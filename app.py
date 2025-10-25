import matplotlib
matplotlib.use('Agg') # Use non-GUI backend
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template, send_from_directory, url_for
import io
import re # Import regex for categorization
import traceback # For detailed error printing
import numpy as np # For conditional logic
import csv # For manual header detection
import pdfplumber # For reading PDFs
import chardet # For encoding detection

# --- Configuration ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_IMAGES_FOLDER = 'static/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_IMAGES_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_IMAGES_FOLDER'] = STATIC_IMAGES_FOLDER

# --- Category Keywords ---
CATEGORY_KEYWORDS = { # Using standard dictionary definition
    'Food & Dining': ['swiggy', 'zomato', 'restaurant', 'cafe', 'food', 'lunch', 'dinner', 'amruttulya', 'mess', 'hotel'],
    'Bills & Utilities': ['jio', 'vi', 'vodafone', 'airtel', 'electricity', 'bill', 'recharge', 'gas'],
    'Shopping': ['store', 'mart', 'market', 'mall', 'shop', 'amazon', 'flipkart', 'myntra'],
    'Travel': ['ola', 'uber', 'bus', 'travel', 'fuel', 'petrol', 'station'],
    'Entertainment': ['movie', 'cinema', 'bookmyshow', 'pvr', 'inox', 'game'],
    'Groceries': ['grocery', 'supermarket', 'bigbasket', 'instamart'],
    'Health & Wellness': ['medical', 'pharmacy', 'hospital', 'doctor'],
    'Transfers': ['transfer', 'sent to', 'received from'],
    'Other': []
}

# --- Categorization Function ---
def categorize_transaction(details_or_payee):
    if not isinstance(details_or_payee, str): return 'Other'
    text = details_or_payee.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', text) for keyword in keywords):
            return category
    return 'Other'

# --- 1. CLEANING & DETECTION FUNCTIONS ---
def clean_col_name(col_name):
    if not isinstance(col_name, str): return ""
    clean_col = str(col_name).lower().replace('\n', '').replace('\r', '').replace('\t', '')
    clean_col = re.sub(r'\s+', '', clean_col) # Remove all spaces
    return clean_col.strip()

# --- MORE FLEXIBLE DETECTOR ---
def detect_file_type(df_columns, file_extension, df_preview=None):
    cleaned_cols = {clean_col_name(col) for col in df_columns}
    print(f"--- DEBUG: Columns for detection: {cleaned_cols} ---")

    # POP UPI Report Check
    if '#' in cleaned_cols or '' in cleaned_cols:
        if 'payervpa' in cleaned_cols and 'receivervpa' in cleaned_cols and 'transactiontimestamp' in cleaned_cols:
            print("Detected: POP_UPI_FORMAT (CSV)")
            return 'POP_UPI_FORMAT'

    # --- PhonePe Statement Check (More Flexible for PDF) ---
    has_date = 'date' in cleaned_cols
    # Check if cleaned name *contains* 'transactiondetails' or 'details' (from PDF)
    has_details = any('transactiondetails' in c or c == 'details' for c in cleaned_cols)
    has_type = 'type' in cleaned_cols
    # Check if cleaned name *starts with* 'amount' (handles 'amount', 'amoun', 'amo')
    has_amount = any(c.startswith('amount') for c in cleaned_cols)

    if has_date and has_details and has_type and has_amount:
        print(f"Detected: PHONEPE_STATEMENT_FORMAT ({file_extension})")
        return 'PHONEPE_STATEMENT_FORMAT'
    # --- End Flexible Check ---

    # HDFC CSV Placeholder
    if 'narration' in cleaned_cols and 'valuedt' in cleaned_cols and file_extension == '.csv':
        print("Detected: HDFC_CSV_FORMAT (Placeholder)")
        return 'HDFC_CSV_FORMAT'

    # ICICI Excel Placeholder
    if 'transactionremarks' in cleaned_cols and 'withdrawalamount(inr)' in cleaned_cols and file_extension in ['.xls', '.xlsx']:
         print("Detected: ICICI_EXCEL_FORMAT (Placeholder)")
         return 'ICICI_EXCEL_FORMAT'

    # Generic PDF Check (Fallback)
    if file_extension == '.pdf':
        has_pdf_date = any(kw in cleaned_cols for kw in ['date', 'valuedt', 'txndate'])
        has_pdf_desc = any(kw in cleaned_cols for kw in ['description', 'narration', 'particulars', 'remarks', 'details'])
        has_pdf_debit = any(kw in cleaned_cols for kw in ['debit', 'withdrawal', 'dr', 'w/d']) # Removed dot
        has_pdf_credit = any(kw in cleaned_cols for kw in ['credit', 'deposit', 'cr', 'dep']) # Removed dot
        if has_pdf_date and (has_pdf_debit or has_pdf_credit):
             print("Detected: GENERIC_PDF_FORMAT")
             return 'GENERIC_PDF_FORMAT'

    print(f"Detected: UNKNOWN_FORMAT.")
    return 'UNKNOWN_FORMAT'

# --- 2. "EXPERT" PROCESSOR FUNCTIONS ---

# --- MORE ROBUST PhonePe Processor (Handles CSV/Excel/PDF-like structure) ---
def process_phonepe_statement_format(df):
    """
    Expert for PhonePe statement format (CSV/Excel/PDF).
    Increased robustness for data cleaning.
    """
    print("Processing with: process_phonepe_statement_format")
    df_processed = df.copy() # Avoid modifying original DataFrame
    print(f"--- DEBUG: Input DF shape: {df_processed.shape} ---")
    print(df_processed.head(3).to_string())

    norm_df = pd.DataFrame()
    original_cols = {clean_col_name(col): col for col in df_processed.columns}

    # Find columns using flexible matching
    date_col = original_cols.get('date')
    # Find details (handles 'Transaction Details' or 'Details' or 'Transaction\nDetails')
    details_col = next((orig for clean, orig in original_cols.items() if 'transactiondetails' in clean or clean == 'details'), None)
    type_col = original_cols.get('type')
    # Find amount (handles 'Amount', 'Amoun', 'Amo')
    amount_col = next((orig for clean, orig in original_cols.items() if clean.startswith('amount')), None)

    if not all([date_col, details_col, type_col, amount_col]):
        missing = []
        if not date_col: missing.append("Date")
        if not details_col: missing.append("Details")
        if not type_col: missing.append("Type")
        if not amount_col: missing.append("Amount")
        raise ValueError(f"PhonePe processor missing required columns: {', '.join(missing)}. Found: {list(df_processed.columns)}")
    
    # --- Handle PDF case where columns are split (e.g., 'Transaction', 'Details') ---
    # This is complex, but for now we rely on the flexible 'details_col' finder.
    # If `details_col` matched just 'Details', we might need to combine it with 'Transaction'.
    # Let's check if 'Transaction' exists *if* 'details_col' is just 'Details'.
    if clean_col_name(details_col) == 'details':
        transaction_col = next((orig for clean, orig in original_cols.items() if clean == 'transaction'), None)
        if transaction_col:
            print("--- DEBUG: Found split 'Transaction' and 'Details' columns. Combining them. ---")
            df_processed[details_col] = df_processed[transaction_col].astype(str) + " " + df_processed[details_col].astype(str)

    # --- Data Cleaning ---
    # Convert ALL columns to string first for robust cleaning
    df_processed[date_col] = df_processed[date_col].astype(str).str.strip()
    df_processed[details_col] = df_processed[details_col].astype(str) # No strip, handle in payee func
    df_processed[type_col] = df_processed[type_col].astype(str).str.strip().str.upper()
    df_processed[amount_col] = df_processed[amount_col].astype(str).str.strip()

    # Drop rows where essential cols are originally NaN/empty *before* conversion
    df_processed = df_processed.dropna(subset=[date_col, type_col, amount_col])
    df_processed = df_processed[df_processed[date_col].str.len() > 0]
    df_processed = df_processed[df_processed[type_col].str.len() > 0]
    df_processed = df_processed[df_processed[amount_col].str.len() > 0]
    if df_processed.empty:
        print("WARN: DataFrame empty after dropping initial NaNs/empty strings.")
        return pd.DataFrame()

    # Clean Date
    print("--- DEBUG: Cleaning Date ---")
    date_cleaned = df_processed[date_col].str.replace(r'[\n\r]+', ' ', regex=True).str.strip()
    try:
        norm_df['Std_Date'] = pd.to_datetime(date_cleaned, format='%b %d, %Y', errors='coerce')
    except Exception:
        norm_df['Std_Date'] = pd.to_datetime(date_cleaned, errors='coerce')

    # Clean Amount
    print("--- DEBUG: Cleaning Amount ---")
    amount_cleaned = df_processed[amount_col].str.replace(r'[^\d.]', '', regex=True)
    amount_numeric = pd.to_numeric(amount_cleaned.replace('', '0'), errors='coerce').fillna(0)

    # Clean and Determine Type
    print("--- DEBUG: Determining Type ---")
    type_cleaned = df_processed[type_col] # Already cleaned
    is_debit = (type_cleaned == 'DEBIT')

    norm_df['Std_Debit'] = amount_numeric.where(is_debit, 0)
    norm_df['Std_Credit'] = amount_numeric.where(~is_debit, 0)

    # Extract Payee/Description
    print("--- DEBUG: Extracting Description ---")
    def get_payee_details(row_details):
        if not isinstance(row_details, str): return 'N/A'
        lines = [line.strip() for line in row_details.split('\n') if line.strip()]
        if not lines: return 'N/A'
        first_line = lines[0]
        if first_line.startswith('Paid to '): return first_line.replace('Paid to ', '').strip()
        if first_line.startswith('Received from '): return first_line.replace('Received from ', '').strip()
        if first_line.startswith('Transfer to '): return first_line
        if 'recharged' in first_line.lower(): return first_line
        if 'bill paid' in first_line.lower(): return first_line
        if len(lines) > 1 and (re.match(r'^\d{1,2}:\d{2}\s*(?:am|pm)$', first_line, re.IGNORECASE) or 'Transaction ID' in first_line or 'UTR No.' in first_line):
            return lines[1]
        return first_line if first_line else 'N/A'

    norm_df['Std_Description'] = df_processed[details_col].apply(get_payee_details)

    norm_df['Std_Type'] = np.where(is_debit, 'DEBIT', 'CREDIT')
    norm_df['Std_Amount'] = amount_numeric

    # --- Final cleanup - CRITICAL STEP ---
    rows_before_drop = len(norm_df)
    print(f"Shape before final drop: {norm_df.shape}")
    norm_df = norm_df.dropna(subset=['Std_Date'])
    print(f"Shape after dropna(Std_Date): {norm_df.shape}")
    norm_df = norm_df[norm_df['Std_Amount'] > 0]
    rows_after_drop = len(norm_df)
    print(f"Shape after drop Amount > 0: {rows_after_drop}")

    if norm_df.empty and rows_before_drop > 0:
         print("WARN: All rows dropped during final cleaning. Check date/amount formats in source.")
    return norm_df


def process_pop_upi_format(df):
    """ Expert function for the POP_UPI_TRANSACTION_REPORT file. """
    print("Processing with: process_pop_upi_format")
    norm_df = pd.DataFrame()
    cleaned_to_original = {clean_col_name(col): col for col in df.columns}
    date_col = cleaned_to_original.get('transactiontimestamp')
    type_col = cleaned_to_original.get('transactiontype')
    amount_col = cleaned_to_original.get('amount')
    payer_col = cleaned_to_original.get('payername')
    receiver_col = cleaned_to_original.get('receivername')
    if not all([date_col, type_col, amount_col, payer_col, receiver_col]): raise ValueError("POP UPI columns missing")
    date_str_cleaned = df[date_col].astype(str).str.replace(r'\s+', '', regex=True)
    norm_df['Std_Date'] = pd.to_datetime(date_str_cleaned, errors='coerce')
    amount_str_cleaned = df[amount_col].astype(str).str.replace(r'[\s,]', '', regex=True)
    amount_numeric = pd.to_numeric(amount_str_cleaned, errors='coerce').fillna(0)
    type_str_cleaned = df[type_col].astype(str).str.replace(r'\s+', '', regex=True).str.upper()
    is_debit = (type_str_cleaned == 'PAY')
    norm_df['Std_Debit'] = amount_numeric.where(is_debit, 0)
    norm_df['Std_Credit'] = amount_numeric.where(~is_debit, 0)
    norm_df['Std_Description'] = np.where(is_debit,
                                        df[receiver_col].astype(str).str.replace(r'[\n\r]+', ' ', regex=True).str.strip(),
                                        df[payer_col].astype(str).str.replace(r'[\n\r]+', ' ', regex=True).str.strip())
    norm_df['Std_Amount'] = amount_numeric
    norm_df['Std_Type'] = np.where(is_debit, 'DEBIT', 'CREDIT')
    rows_before_drop = len(norm_df); norm_df = norm_df.dropna(subset=['Std_Date']); norm_df = norm_df[(norm_df['Std_Debit'] > 0) | (norm_df['Std_Credit'] > 0)]; rows_after_drop = len(norm_df)
    print(f"Rows before final drop: {rows_before_drop}, Rows after: {rows_after_drop}")
    return norm_df

# --- Placeholders ---
def process_hdfc_csv_format(df): raise NotImplementedError("HDFC CSV")
def process_icici_excel_format(df): raise NotImplementedError("ICICI Excel")

# --- Generic PDF Processor (Fallback) ---
def process_generic_pdf_format(df):
    """ Attempts to process a DataFrame extracted from a PDF (Fallback). """
    print("Processing with: process_generic_pdf_format (Fallback)")
    norm_df = pd.DataFrame()
    cleaned_to_original = {clean_col_name(c): c for c in df.columns}
    date_col = next((orig for clean, orig in cleaned_to_original.items() if any(k in clean for k in ['date', 'valuedt', 'txndt'])), None)
    desc_col = next((orig for clean, orig in cleaned_to_original.items() if any(k in clean for k in ['description', 'narration', 'particulars', 'remarks', 'details'])), None)
    debit_col = next((orig for clean, orig in cleaned_to_original.items() if any(k in clean for k in ['debit', 'withdrawal', 'dr', 'w/d'])), None)
    credit_col = next((orig for clean, orig in cleaned_to_original.items() if any(k in clean for k in ['credit', 'deposit', 'cr', 'dep'])), None)
    if not (date_col and (debit_col or credit_col)):
        missing = [];
        if not date_col: missing.append("Date")
        if not (debit_col or credit_col): missing.append("Debit/Credit")
        raise ValueError(f"Generic PDF missing: {', '.join(missing)}. Columns: {list(df.columns)}")
    if not desc_col: df['__temp_desc__'] = 'N/A'; desc_col = '__temp_desc__'; print("WARN: Using 'N/A' for missing PDF description.")
    norm_df['Std_Date'] = pd.to_datetime(df[date_col].astype(str).str.strip(), errors='coerce')
    if debit_col and debit_col in df.columns:
        debit_str_cleaned = df[debit_col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        norm_df['Std_Debit'] = pd.to_numeric(debit_str_cleaned, errors='coerce').fillna(0)
    else: norm_df['Std_Debit'] = 0.0
    if credit_col and credit_col in df.columns:
        credit_str_cleaned = df[credit_col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        norm_df['Std_Credit'] = pd.to_numeric(credit_str_cleaned, errors='coerce').fillna(0)
    else: norm_df['Std_Credit'] = 0.0
    norm_df['Std_Description'] = df[desc_col].astype(str).str.replace(r'[\n\r]+', ' ', regex=True).str.strip()
    is_debit = norm_df['Std_Debit'] > 0
    norm_df['Std_Amount'] = norm_df['Std_Debit'] + norm_df['Std_Credit']
    norm_df['Std_Type'] = np.where(is_debit, 'DEBIT', 'CREDIT')
    rows_before_drop = len(norm_df); norm_df = norm_df.dropna(subset=['Std_Date']); norm_df = norm_df[norm_df['Std_Amount'] > 0]; rows_after_drop = len(norm_df)
    print(f"Rows before final drop (Generic PDF): {rows_before_drop}, Rows after: {rows_after_drop}")
    return norm_df


# --- 4. HELPER TO FIND CSV HEADER ---
def find_csv_header(file_stream, encoding='utf-8'):
    keywords = {'date', 'transaction', 'type', 'amount'}
    max_lines_to_check = 10; potential_header = -1
    try:
        peek_data = file_stream.peek(4096); text_stream = io.TextIOWrapper(io.BytesIO(peek_data), encoding=encoding, errors='replace'); reader = csv.reader(text_stream)
        for i, row in enumerate(reader):
            if i >= max_lines_to_check: break
            if not row: continue
            row_content_lower = {str(cell).lower().strip() for cell in row}
            matches = 0
            if 'date' in row_content_lower: matches += 1
            if 'transaction details' in row_content_lower: matches +=1
            if 'type' in row_content_lower: matches += 1
            if 'amount' in row_content_lower: matches += 1
            if matches >= 4: potential_header = i; print(f"--- Found PhonePe CSV header on row {i} ---"); break
    except Exception as e: print(f"Error finding header: {e}"); potential_header = -1
    file_stream.seek(0)
    if potential_header == -1: print("--- PhonePe CSV header not found automatically. ---"); return None
    else: return potential_header


# --- 5. ENHANCED PDF READING FUNCTION ---
def read_pdf_tables(file_stream):
    """ Extracts tables using pdfplumber, with more debugging. """
    potential_dfs = []
    try:
        file_stream.seek(0); initial_bytes = file_stream.read(1024); file_stream.seek(0)
        if b'/Encrypt' in initial_bytes: raise ValueError("Cannot process password-protected PDF files.")

        with pdfplumber.open(file_stream) as pdf:
            print(f"Opened PDF with {len(pdf.pages)} pages.")
            settings_to_try = [ {}, {"vertical_strategy": "text", "horizontal_strategy": "text"}, {"vertical_strategy": "lines", "horizontal_strategy": "lines"} ]
            
            for settings_idx, settings in enumerate(settings_to_try):
                print(f"\n--- Trying PDF strategy #{settings_idx+1}: {settings} ---")
                page_dfs_current_strategy = []
                for i, page in enumerate(pdf.pages):
                    try:
                        tables = page.extract_tables(table_settings=settings)
                        print(f"Page {i+1}: Found {len(tables)} tables.")
                        for table_index, table in enumerate(tables):
                            if table and len(table) > 1:
                                header = table[0]; data = table[1:]
                                cleaned_header = [(str(h).replace('\n', ' ').strip() if h is not None else f"Column_{j+1}") for j, h in enumerate(header)]
                                if len([h for h in cleaned_header if h.startswith('Column_')]) > len(cleaned_header) / 2:
                                    print(f"   Skipping table {table_index+1} on page {i+1} (bad header): {cleaned_header}")
                                    continue
                                df = pd.DataFrame(data, columns=cleaned_header)
                                print(f"\n   DEBUG PDF Table Content (Page {i+1}, Table {table_index+1}, Strategy {settings_idx+1}):")
                                print(df.head(3).to_string()) # Print first 3 rows
                                print("-" * 30)
                                page_dfs_current_strategy.append(df)
                    except Exception as page_ex: print(f"   Error extracting tables from page {i+1}: {page_ex}"); continue
                if page_dfs_current_strategy: 
                    potential_dfs.extend(page_dfs_current_strategy)
                    # Don't break, try all strategies to get all possible tables

    except ValueError as ve: raise ve
    except Exception as pdf_e: raise ValueError(f"Failed to read/parse PDF: {pdf_e}")

    # --- Analyze all potential DataFrames ---
    likely_transaction_tables = []; best_match_score = -1
    if not potential_dfs: raise ValueError("No tables extracted from PDF. File might be scanned, image-based, or have unsupported layout.")

    print(f"\n--- Analyzing {len(potential_dfs)} extracted tables ---")
    for df_idx, df in enumerate(potential_dfs):
        if df is None or df.empty: continue
        print(f"Analyzing Table {df_idx+1} (cols: {list(df.columns)})")
        cols_lower_cleaned = {clean_col_name(c): c for c in df.columns}
        
        # --- Flexible Column Finding ---
        date_col = next((orig for clean, orig in cols_lower_cleaned.items() if any(k in clean for k in ['date', 'valuedt', 'txndt'])), None)
        desc_col = next((orig for clean, orig in cols_lower_cleaned.items() if any(k in clean for k in ['description', 'narration', 'particulars', 'remarks', 'details'])), None)
        debit_col = next((orig for clean, orig in cols_lower_cleaned.items() if any(k in clean for k in ['debit', 'withdrawal', 'dr', 'w/d'])), None)
        credit_col = next((orig for clean, orig in cols_lower_cleaned.items() if any(k in clean for k in ['credit', 'deposit', 'cr', 'dep'])), None)
        type_col = next((orig for clean, orig in cols_lower_cleaned.items() if clean == 'type'), None)
        amount_col = next((orig for clean, orig in cols_lower_cleaned.items() if clean.startswith('amount')), None)

        score = 0
        is_phonepe_like = False
        if date_col and desc_col and type_col and amount_col:
             score = 10 # High score for exact PhonePe-like match
             is_phonepe_like = True
             print("    --> Looks like PhonePe PDF format")
        else: # Score based on generic debit/credit columns
            if date_col: score += 3
            if desc_col: score += 2
            if debit_col: score += 1
            if credit_col: score += 1

        print(f"   Score: {score}. Found: Date='{date_col}', Desc='{desc_col}', Debit='{debit_col}', Credit='{credit_col}', Type='{type_col}', Amount='{amount_col}'")

        # Require at least Date and (Debit or Credit or Amount)
        if score >= 3 and date_col and (debit_col or credit_col or amount_col):
             if score > best_match_score:
                 print(f"   *** New best candidate table found (Score: {score}) ***")
                 likely_transaction_tables = [df] # Replace previous best
                 best_match_score = score
             elif score == best_match_score:
                 print(f"   Adding as candidate table (Score: {score})")
                 likely_transaction_tables.append(df)

    if not likely_transaction_tables:
         raise ValueError("Could not identify a table with required columns (Date, Debit/Credit/Amount) in the PDF.")

    print(f"--- Identified {len(likely_transaction_tables)} likely transaction table(s) ---")
    return likely_transaction_tables


# --- 6. MAIN APP ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return render_template('index.html', error="No file part.")
    file = request.files['file']
    if file.filename == '': return render_template('index.html', error="No file selected.")

    allowed_extensions = ('.xls', '.xlsx', '.csv', '.pdf')
    file_extension = os.path.splitext(file.filename)[1].lower()
    if not file.filename.endswith(allowed_extensions):
        return render_template('index.html', error=f"Invalid file type. Please upload {', '.join(allowed_extensions)} file.")

    try:
        # --- 1. Read File ---
        all_sheets_or_tables = {}

        if file_extension == '.csv':
            try:
                file.stream.seek(0); initial_bytes = file.stream.peek(5000)
                try: detected_encoding = chardet.detect(initial_bytes)['encoding'] or 'utf-8'
                except Exception: detected_encoding = 'utf-8' # Fallback
                file.stream.seek(0)
                print(f"CSV Encoding: {detected_encoding}")
                header_row_index = find_csv_header(file.stream, encoding=detected_encoding)
                read_header = 0
                if header_row_index is not None: read_header = header_row_index
                elif 'phonepe_statement' in file.filename.lower(): read_header = 2
                print(f"Reading CSV: header={read_header}, encoding={detected_encoding}")
                file.stream.seek(0)
                df_sheet = pd.read_csv(file.stream, comment=None, header=read_header, engine='python', skipinitialspace=True, encoding=detected_encoding, on_bad_lines='warn')
                all_sheets_or_tables['csv_sheet_1'] = df_sheet
            except Exception as read_e:
                print(f"Primary CSV read failed: {read_e}. Trying latin1...")
                try:
                    file.stream.seek(0); header_row_index = find_csv_header(file.stream, encoding='latin1')
                    read_header = header_row_index if header_row_index is not None else (2 if 'phonepe_statement' in file.filename.lower() else 0)
                    file.stream.seek(0)
                    df_sheet = pd.read_csv(file.stream, comment=None, header=read_header, engine='python', skipinitialspace=True, encoding='latin1', on_bad_lines='warn')
                    all_sheets_or_tables['csv_sheet_1'] = df_sheet
                except Exception as fallback_e: raise ValueError(f"Error reading CSV (tried utf-8/latin1): {fallback_e}")

        elif file_extension in ['.xls', '.xlsx']:
            try: file.stream.seek(0); all_sheets_or_tables = pd.read_excel(file, sheet_name=None)
            except Exception as read_e: raise ValueError(f"Error reading Excel: {read_e}")

        elif file_extension == '.pdf':
             try:
                 file.stream.seek(0)
                 pdf_dfs = read_pdf_tables(file.stream) # Use the enhanced function
                 if not pdf_dfs: raise ValueError("No usable transaction tables found in PDF.")
                 for i, pdf_df in enumerate(pdf_dfs): all_sheets_or_tables[f'pdf_table_{i+1}'] = pdf_df
             except Exception as read_e: raise ValueError(f"{read_e}") # Pass the specific error up

        # --- 2. Detect and Process ---
        all_dfs_normalized = []
        if not all_sheets_or_tables: raise ValueError("File read failed or no data found.")

        for name, df_sheet in all_sheets_or_tables.items():
            if df_sheet is None or df_sheet.empty or df_sheet.columns.empty: continue
            df_sheet.columns = df_sheet.columns.astype(str)
            file_type = detect_file_type(df_sheet.columns, file_extension, df_sheet.head())

            try:
                norm_df = None
                if file_type == 'POP_UPI_FORMAT': norm_df = process_pop_upi_format(df_sheet)
                elif file_type == 'PHONEPE_STATEMENT_FORMAT': norm_df = process_phonepe_statement_format(df_sheet)
                elif file_type == 'HDFC_CSV_FORMAT': norm_df = process_hdfc_csv_format(df_sheet)
                elif file_type == 'ICICI_EXCEL_FORMAT': norm_df = process_icici_excel_format(df_sheet)
                elif file_type == 'GENERIC_PDF_FORMAT': norm_df = process_generic_pdf_format(df_sheet)
                else: print(f"Skipping '{name}': No processor for UNKNOWN format."); continue

                if norm_df is not None and not norm_df.empty:
                    print(f"OK: Processed '{name}' as {file_type}. Rows: {len(norm_df)}")
                    all_dfs_normalized.append(norm_df)
                else: print(f"WARN: Processor for '{file_type}' returned empty DF for '{name}'.")

            except NotImplementedError as nie: print(f"WARN: Skipping '{name}': Processor for '{file_type}' not implemented. {nie}"); continue
            except Exception as process_e: raise ValueError(f"Error processing '{name}' (as {file_type}): {process_e}")

        if not all_dfs_normalized:
            return render_template('index.html', error="Could not process any tables/sheets. Check if the file format is supported and contains valid data.")

        # --- 3. Concatenate and Analyze ---
        df = pd.concat(all_dfs_normalized, ignore_index=True)
        df['Std_Date'] = pd.to_datetime(df['Std_Date'], errors='coerce')
        df = df.dropna(subset=['Std_Date'])
        if df.empty: return render_template('index.html', error="No valid dates found after processing.")
        df = df.sort_values(by='Std_Date')
        if df.empty: return render_template('index.html', error="No valid data after cleaning.")

        start_date = df['Std_Date'].min().strftime('%B %d, %Y')
        end_date = df['Std_Date'].max().strftime('%B %d, %Y')

        # --- 4. Analyses ---
        df['Category'] = df['Std_Description'].apply(categorize_transaction)
        top_10_spending = pd.DataFrame()
        spending_df = df[(df['Std_Type'] == 'DEBIT') & (df['Std_Description'] != 'N/A') & (~df['Std_Description'].str.startswith('Transfer to', na=False))]
        if not spending_df.empty: top_10_spending = spending_df.groupby('Std_Description')['Std_Debit'].sum().nlargest(10).reset_index().rename(columns={'Std_Description': 'PayeeOrDetails', 'Std_Debit': 'Amount'})
        monthly_summary = pd.DataFrame()
        df['Month'] = pd.to_datetime(df['Std_Date'], errors='coerce').dt.to_period('M'); monthly_summary = df.groupby('Month')[['Std_Debit', 'Std_Credit']].sum().reset_index(); monthly_summary = monthly_summary.rename(columns={'Std_Debit': 'Debit', 'Std_Credit': 'Credit'})
        if not monthly_summary.empty: monthly_summary['MonthStr'] = monthly_summary['Month'].astype(str)
        total_debit = df['Std_Debit'].sum(); total_credit = df['Std_Credit'].sum()
        day_of_week_spending = pd.Series(dtype=float); df['Day of Week'] = pd.to_datetime(df['Std_Date'], errors='coerce').dt.day_name(); day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']; day_of_week_spending = df[df['Std_Type'] == 'DEBIT'].groupby('Day of Week')['Std_Debit'].sum().reindex(day_order).fillna(0)
        category_spending = pd.Series(dtype=float); category_spending_filtered = df[(df['Std_Type'] == 'DEBIT') & (df['Category'].notna()) & (df['Category'] != 'Other')]; category_spending = category_spending_filtered.groupby('Category')['Std_Debit'].sum().sort_values(ascending=False)
        largest_debits_list, largest_credits_list = [], []
        if total_debit > 0:
            largest_debits = df[df['Std_Debit'] > 0].nlargest(5, 'Std_Debit')[['Std_Date', 'Std_Description', 'Std_Debit']]; largest_debits_list = largest_debits.to_dict('records')
            for item in largest_debits_list: item['Date'] = item['Std_Date'].strftime('%b %d, %Y') if pd.notna(item['Std_Date']) else 'Invalid'; item['PayeeOrDetails'] = item['Std_Description']; item['Debit'] = f"₹{item['Std_Debit']:,.2f}"
        if total_credit > 0:
            largest_credits = df[df['Std_Credit'] > 0].nlargest(5, 'Std_Credit')[['Std_Date', 'Std_Description', 'Std_Credit']]; largest_credits_list = largest_credits.to_dict('records')
            for item in largest_credits_list: item['Date'] = item['Std_Date'].strftime('%b %d, %Y') if pd.notna(item['Std_Date']) else 'Invalid'; item['PayeeOrDetails'] = item['Std_Description']; item['Credit'] = f"₹{item['Std_Credit']:,.2f}"

        # --- 5. Prepare Excel Summary ---
        summary_filename = 'summary_report_with_graphs.xlsx'; summary_path = os.path.join(app.config['UPLOAD_FOLDER'], summary_filename); graph_paths = {}

        # --- 6. Generate Graphs ---
        # (Graph code unchanged)
        category_pie_filename, category_pie_url, category_pie_path = 'category_spending_pie.png', None, None
        if not category_spending.empty and category_spending.sum() > 0:
            plt.figure(figsize=(10, 8)); threshold = category_spending.sum() * 0.02; small_categories = category_spending[category_spending < threshold] if threshold > 0 else pd.Series(dtype=float); large_categories = category_spending[category_spending >= threshold]; plot_data = large_categories.copy()
            if not small_categories.empty: other_sum = small_categories.sum(); plot_data['Other (<2%)'] = other_sum if other_sum > 0 else 0
            plot_data = plot_data[plot_data > 0].sort_values(ascending=False)
            if not plot_data.empty:
                pie_labels = [f'{cat}\n(₹{amt:,.0f})' for cat, amt in plot_data.items()]; colors = plt.cm.viridis(range(len(plot_data)))
                plt.pie(plot_data, labels=pie_labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85, textprops={'color': 'white', 'fontsize': 10, 'fontweight':'bold'}, wedgeprops={'edgecolor': 'white', 'linewidth': 0.5})
                centre_circle = plt.Circle((0,0),0.70,fc='none'); fig = plt.gcf(); fig.gca().add_artist(centre_circle); plt.title('Spending Distribution by Category (Excluding "Other")', color='white', fontsize=16, pad=20); plt.tight_layout()
                category_pie_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], category_pie_filename); plt.savefig(category_pie_path, transparent=True, bbox_inches='tight'); category_pie_url = f'images/{category_pie_filename}'
                if os.path.exists(category_pie_path): graph_paths['category'] = category_pie_path
            plt.close()

        graph_filename, graph_url, graph_path = 'monthly_summary.png', None, None
        if monthly_summary is not None and not monthly_summary.empty:
            monthly_summary_melted_bar = monthly_summary.melt('MonthStr', var_name='Type', value_name='Amount', value_vars=['Debit', 'Credit']); plt.figure(figsize=(10, 6))
            ax = sns.barplot(data=monthly_summary_melted_bar, x='MonthStr', y='Amount', hue='Type', palette={'Debit': '#E74C3C', 'Credit': '#2ECC71'})
            ax.set_title('Monthly Debit vs. Credit', color='white', fontsize=16, pad=20); ax.tick_params(axis='x', colors='white', rotation=45); ax.tick_params(axis='y', colors='white')
            ax.set_xlabel('Month', color='white', fontsize=12); ax.set_ylabel('Amount (₹)', color='white', fontsize=12); ax.legend(title_fontsize='13', labelcolor='white'); ax.set_facecolor('none')
            for p in ax.patches: height = p.get_height(); ax.annotate(f'₹{height:,.0f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', color='white', fontsize=10) if pd.notna(height) and height > 0 else None
            plt.tight_layout(); graph_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], graph_filename); plt.savefig(graph_path, transparent=True, bbox_inches='tight'); graph_url = f'images/{graph_filename}'
            if os.path.exists(graph_path): graph_paths['monthly_bar'] = graph_path
            plt.close()

        cum_graph_filename, cum_graph_url, cum_graph_path = 'cumulative_balance.png', None, None
        if not df.empty and 'Std_Credit' in df.columns and 'Std_Debit' in df.columns:
            df_cb = df.dropna(subset=['Std_Date']).sort_values(by='Std_Date')
            if not df_cb.empty:
                 df_cb['Balance'] = df_cb['Std_Credit'].cumsum() - df_cb['Std_Debit'].cumsum(); plt.figure(figsize=(10, 6))
                 ax = sns.lineplot(data=df_cb, x='Std_Date', y='Balance', color='#00f2ea', linewidth=2.5)
                 ax.set_title('Cumulative Balance Over Time', color='white', fontsize=16, pad=20); ax.tick_params(axis='x', colors='white', rotation=45); ax.tick_params(axis='y', colors='white')
                 ax.set_xlabel('Date', color='white', fontsize=12); ax.set_ylabel('Balance (₹)', color='white', fontsize=12); ax.grid(True, linestyle='--', alpha=0.3, color='white'); ax.set_facecolor('none')
                 plt.tight_layout(); cum_graph_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], cum_graph_filename); plt.savefig(cum_graph_path, transparent=True, bbox_inches='tight'); cum_graph_url = f'images/{cum_graph_filename}'
                 if os.path.exists(cum_graph_path): graph_paths['cumulative'] = cum_graph_path
                 plt.close()

        top_10_graph_filename, top_10_graph_url, top_10_graph_path = 'top_10_spending.png', None, None
        if top_10_spending is not None and not top_10_spending.empty:
            plt.figure(figsize=(10, 8)); ax = sns.barplot(data=top_10_spending, y='PayeeOrDetails', x='Amount', palette='viridis_r')
            ax.set_title('Top 10 Spending (by Payee/Details)', color='white', fontsize=16, pad=20); ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white', labelsize=10)
            ax.set_xlabel('Total Amount (₹)', color='white', fontsize=12); ax.set_ylabel('Payee / Details', color='white', fontsize=12); ax.set_facecolor('none')
            for i, (value, name) in enumerate(zip(top_10_spending['Amount'], top_10_spending['PayeeOrDetails'])): ax.text(value + 1, i, f' ₹{value:,.0f}', va='center', ha='left', color='white', fontsize=10) if pd.notna(value) else None
            plt.tight_layout(); top_10_graph_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], top_10_graph_filename); plt.savefig(top_10_graph_path, transparent=True, bbox_inches='tight'); plt.close(); top_10_graph_url = f'images/{top_10_graph_filename}'
            if os.path.exists(top_10_graph_path): graph_paths['top_10'] = top_10_graph_path

        pie_chart_filename, pie_chart_url, pie_chart_path = 'debit_credit_pie.png', None, None
        if total_debit > 0 or total_credit > 0:
            pie_data = [total_debit, total_credit]; pie_labels = [f'Total Debit\n(₹{total_debit:,.0f})', f'Total Credit\n(₹{total_credit:,.0f})']; colors = ['#E74C3C', '#2ECC71']; plt.figure(figsize=(8, 8))
            plt.pie(pie_data, labels=pie_labels, colors=colors, autopct='%1.1f%%', textprops={'color': 'white', 'fontsize': 12, 'fontweight': 'bold'}, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
            plt.title('Total Debit vs. Credit', color='white', fontsize=16, pad=20); plt.tight_layout(); pie_chart_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], pie_chart_filename); plt.savefig(pie_chart_path, transparent=True, bbox_inches='tight'); plt.close(); pie_chart_url = f'images/{pie_chart_filename}'
            if os.path.exists(pie_chart_path): graph_paths['debit_credit'] = pie_chart_path

        day_of_week_filename, day_of_week_url, day_of_week_path = 'day_of_week_spending.png', None, None
        day_of_week_spending_filled = day_of_week_spending.fillna(0)
        if not day_of_week_spending_filled.eq(0).all():
            plt.figure(figsize=(10, 6)); ax = sns.barplot(x=day_of_week_spending_filled.index, y=day_of_week_spending_filled.values, palette='plasma')
            ax.set_title('Spending by Day of Week', color='white', fontsize=16, pad=20); ax.tick_params(axis='x', colors='white', rotation=45); ax.tick_params(axis='y', colors='white')
            ax.set_xlabel('Day of Week', color='white', fontsize=12); ax.set_ylabel('Total Amount (₹)', color='white', fontsize=12); ax.set_facecolor('none')
            for p in ax.patches: height = p.get_height(); ax.annotate(f'₹{height:,.0f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', color='white', fontsize=10) if pd.notna(height) and height > 0 else None
            plt.tight_layout(); day_of_week_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], day_of_week_filename); plt.savefig(day_of_week_path, transparent=True, bbox_inches='tight'); plt.close(); day_of_week_url = f'images/{day_of_week_filename}'
            if os.path.exists(day_of_week_path): graph_paths['day_of_week'] = day_of_week_path

        trend_filename, trend_url, trend_path = 'income_expense_trend.png', None, None
        if monthly_summary is not None and not monthly_summary.empty and 'Month' in monthly_summary.columns:
            monthly_summary['MonthDate'] = monthly_summary['Month'].dt.to_timestamp(); plt.figure(figsize=(10, 6))
            plt.plot(monthly_summary['MonthDate'], monthly_summary['Credit'], marker='o', linestyle='-', color='#2ECC71', linewidth=2.5, label='Income (Credit)')
            plt.plot(monthly_summary['MonthDate'], monthly_summary['Debit'], marker='o', linestyle='-', color='#E74C3C', linewidth=2.5, label='Expenses (Debit)')
            plt.title('Monthly Income vs. Expense Trend', color='white', fontsize=16, pad=20); plt.xlabel('Month', color='white', fontsize=12); plt.ylabel('Amount (₹)', color='white', fontsize=12)
            plt.xticks(rotation=45, color='white'); plt.yticks(color='white'); plt.grid(True, linestyle='--', alpha=0.3, color='white'); plt.legend(labelcolor='white'); plt.gca().set_facecolor('none'); plt.tight_layout()
            trend_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], trend_filename); plt.savefig(trend_path, transparent=True, bbox_inches='tight'); plt.close(); trend_url = f'images/{trend_filename}'
            if os.path.exists(trend_path): graph_paths['trend'] = trend_path


        # --- 7. Save Excel Summary ---
        excel_write_successful = False
        if monthly_summary is not None and not monthly_summary.empty:
            try:
                with pd.ExcelWriter(summary_path, engine='xlsxwriter') as writer:
                    summary_df_to_write = monthly_summary[['MonthStr', 'Debit', 'Credit']].rename(columns={'MonthStr':'Month'}); summary_df_to_write.to_excel(writer, index=False, sheet_name='Summary', startrow=0)
                    workbook = writer.book; worksheet = writer.sheets['Summary']; insert_row = len(summary_df_to_write) + 5; image_options = {'x_scale': 0.5, 'y_scale': 0.5}
                    image_order = ['category', 'debit_credit', 'trend', 'day_of_week', 'monthly_bar', 'cumulative', 'top_10']; row_heights = {'category': 30, 'debit_credit': 30, 'trend': 25, 'day_of_week': 25, 'monthly_bar': 25, 'cumulative': 25, 'top_10': 30}
                    col_num, max_col_images, current_col_count = 0, 2, 0
                    for key in image_order:
                        if key in graph_paths and graph_paths[key]:
                            cell = f"{chr(ord('A') + col_num * 10)}{insert_row}"; worksheet.insert_image(cell, graph_paths[key], image_options)
                            current_col_count += 1
                            if current_col_count >= max_col_images: col_num, current_col_count = 0, 0; insert_row += row_heights.get(key, 25) + 2
                            else: col_num += 1
                excel_write_successful = True
            except Exception as excel_error:
                print(f"Error writing Excel: {excel_error}")
                try: # Fallback
                    with pd.ExcelWriter(summary_path, engine='xlsxwriter') as writer: summary_df_to_write = monthly_summary[['MonthStr', 'Debit', 'Credit']].rename(columns={'MonthStr':'Month'}); summary_df_to_write.to_excel(writer, index=False, sheet_name='Summary', startrow=0)
                    print("Saved Excel summary without images."); excel_write_successful = True
                except Exception as fallback_e: print(f"Fallback Excel write failed: {fallback_e}")

        download_file_to_pass = summary_filename if excel_write_successful else None

        # --- 8. Render Results ---
        return render_template('index.html',
                                category_pie_url=category_pie_url, graph_url=graph_url, cum_graph_url=cum_graph_url,
                                top_10_graph_url=top_10_graph_url, pie_chart_url=pie_chart_url, day_of_week_url=day_of_week_url,
                                trend_url=trend_url, download_filename=download_file_to_pass, start_date=start_date,
                                end_date=end_date, largest_debits=largest_debits_list, largest_credits=largest_credits_list)

    except ValueError as ve: # Catch specific ValueErrors
         print(f"Data processing/reading error: {ve}"); traceback.print_exc()
         return render_template('index.html', error=f"{ve}")
    except Exception as e: # Catch all other errors
        print(f"Unexpected error: {e}"); traceback.print_exc()
        return render_template('index.html', error=f"An unexpected error occurred: {e}. Please ensure the file format is supported.")


@app.route('/download/<filename>')
def download_file(filename):
    if filename != 'summary_report_with_graphs.xlsx': return "Invalid filename", 400
    try: return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError: return "Summary file not found.", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=True)