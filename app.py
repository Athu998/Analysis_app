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
import chardet # Library to detect encoding

# --- Configuration ---
app = Flask(__name__)

# Create directories if they don't exist
UPLOAD_FOLDER = 'uploads'
STATIC_IMAGES_FOLDER = 'static/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_IMAGES_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_IMAGES_FOLDER'] = STATIC_IMAGES_FOLDER

# Category Keywords (Customize these!)
CATEGORY_KEYWORDS = {
    'Food & Dining': ['swiggy', 'zomato', 'restaurant', 'cafe', 'food', 'lunch', 'dinner', 'amruttulya', 'mess', 'hotel'],
    'Bills & Utilities': ['jio', 'vi', 'vodafone', 'airtel', 'electricity', 'bill', 'recharge', 'gas'],
    'Shopping': ['store', 'mart', 'market', 'mall', 'shop', 'amazon', 'flipkart', 'myntra'],
    'Travel': ['ola', 'uber', 'bus', 'travel', 'fuel', 'petrol', 'station'],
    'Entertainment': ['movie', 'cinema', 'bookmyshow', 'pvr', 'inox', 'game'],
    'Groceries': ['grocery', 'supermarket', 'bigbasket', 'instamart'],
    'Health & Wellness': ['medical', 'pharmacy', 'hospital', 'doctor'],
    'Transfers': ['transfer', 'sent to', 'received from'],
    'Other': [] # Default category
}

# Categorization Function
def categorize_transaction(details_or_payee):
    if not isinstance(details_or_payee, str):
        return 'Other'
    text = details_or_payee.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        # Use word boundaries for better matching
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', text) for keyword in keywords):
            return category
    return 'Other'

# --- 1. AGGRESSIVE CLEANING FUNCTION ---
def clean_col_name(col_name):
    """
    Aggressively cleans column names to be lowercase and have NO spaces.
    e.g., "Transac tion\nTimesta mp" -> "transactiontimestamp"
    """
    if not isinstance(col_name, str):
        return ""
    # Make lowercase, remove all newlines, tabs, and carriage returns
    clean_col = str(col_name).lower().replace('\n', '').replace('\r', '').replace('\t', '')
    # Remove all whitespace (spaces)
    clean_col = re.sub(r'\s+', '', clean_col)
    return clean_col.strip()
    
# --- 2. "DETECTOR" FUNCTION ---
def detect_file_type(df_columns):
    """
    Analyzes *cleaned* column names to detect the file type.
    """
    cleaned_cols = {clean_col_name(col) for col in df_columns}
    print(f"--- DEBUG: Columns for detection: {cleaned_cols} ---") 
    
    # --- Detection Logic for POP UPI Report ---
    if 'payervpa' in cleaned_cols and 'receivervpa' in cleaned_cols and 'transactiontimestamp' in cleaned_cols:
        print("Detected: POP_UPI_FORMAT")
        return 'POP_UPI_FORMAT'
        
    # --- Detection Logic for (New) PhonePe Statement File ---
    # This file has 'Transaction Details' and is missing VPA columns.
    if 'transactiondetails' in cleaned_cols and 'date' in cleaned_cols and 'type' in cleaned_cols and 'amount' in cleaned_cols:
        print("Detected: PHONEPE_STATEMENT_FORMAT") # Use a new name
        return 'PHONEPE_STATEMENT_FORMAT'

    print(f"Detected: UNKNOWN_FORMAT.")
    return 'UNKNOWN_FORMAT'

# --- 3. "EXPERT" PROCESSOR FUNCTIONS ---

# === NEW NAME and UPDATED LOGIC for the statement file ===
def process_phonepe_statement_format(df):
    """
    Expert function for the newer PhonePe statement format
    (where header is on row 3 and amount has symbols).
    """
    print("Processing with: process_phonepe_statement_format")
    norm_df = pd.DataFrame()
    
    # Find columns using the cleaning logic
    col_map = {}
    # Use original column names directly as pandas read them correctly (due to header=2)
    original_cols = {clean_col_name(col): col for col in df.columns} 
    
    date_col = original_cols.get('date')
    details_col = original_cols.get('transactiondetails')
    type_col = original_cols.get('type')
    amount_col = original_cols.get('amount')
    
    if not all([date_col, details_col, type_col, amount_col]):
        missing = [k for k, v in locals().items() if v is None and k.endswith('_col')]
        raise ValueError(f"PhonePe Statement file missing required columns. Could not find: {missing}")

    # --- Start of logic specific to this format ---
    # Drop rows where essential original columns are missing BEFORE cleaning
    df = df.dropna(subset=[date_col, type_col, amount_col])

    # Clean Date (seems okay, but use fallback)
    try:
        norm_df['Std_Date'] = pd.to_datetime(df[date_col], format='%b %d, %Y', errors='raise')
    except (ValueError, TypeError):
        norm_df['Std_Date'] = pd.to_datetime(df[date_col], errors='coerce') 

    # Clean Amount (remove ₹ and ,)
    amount_str_cleaned = df[amount_col].astype(str).str.replace(r'[₹,]', '', regex=True)
    amount_numeric = pd.to_numeric(amount_str_cleaned, errors='coerce').fillna(0)
    
    # Determine Type and create Debit/Credit
    df[type_col] = df[type_col].astype(str).str.upper() # Ensure uppercase for comparison
    is_debit = (df[type_col] == 'DEBIT')
    is_credit = (df[type_col] == 'CREDIT')

    norm_df['Std_Debit'] = amount_numeric.where(is_debit, 0)
    norm_df['Std_Credit'] = amount_numeric.where(is_credit, 0)

    # Extract Payee/Description from multi-line details
    def get_payee_details(row_details):
        if not isinstance(row_details, str):
            return 'N/A'
        lines = row_details.split('\n')
        first_line = lines[0].strip()
        
        # Specific patterns for this format
        if first_line.startswith('Paid to '):
            return first_line.replace('Paid to ', '').strip()
        elif first_line.startswith('Received from '):
             return first_line.replace('Received from ', '').strip()
        elif first_line.startswith('Transfer to '):
             return first_line # Keep 'Transfer to XXXXX' as description
        # Add more patterns if needed (e.g., 'Mobile recharged')
        elif 'recharged' in first_line.lower():
             return first_line # Keep 'Mobile recharged XXXX'
        elif 'bill paid' in first_line.lower():
             return first_line # Keep 'Electricity bill paid XXXX'
        
        # If no specific pattern, return the first line as description
        return first_line if first_line else 'N/A'

    norm_df['Std_Description'] = df[details_col].apply(get_payee_details)
    # --- End of specific logic ---

    norm_df['Std_Type'] = np.where(is_debit, 'DEBIT', 'CREDIT')
    norm_df['Std_Amount'] = amount_numeric
    
    # Final cleanup
    norm_df = norm_df.dropna(subset=['Std_Date']) # Drop rows where date conversion failed
    norm_df = norm_df[norm_df['Std_Amount'] > 0] # Drop rows with zero amount
    return norm_df

def process_pop_upi_format(df):
    """
    Expert function for the POP_UPI_TRANSACTION_REPORT file.
    (This function should be correct now)
    """
    print("Processing with: process_pop_upi_format")
    norm_df = pd.DataFrame()
    
    col_map = {}
    cleaned_to_original = {clean_col_name(col): col for col in df.columns}

    date_col = cleaned_to_original.get('transactiontimestamp')
    type_col = cleaned_to_original.get('transactiontype')
    amount_col = cleaned_to_original.get('amount')
    payer_col = cleaned_to_original.get('payername')
    receiver_col = cleaned_to_original.get('receivername')
            
    if not all([date_col, type_col, amount_col, payer_col, receiver_col]):
        missing = [k for k,v in locals().items() if v is None and k.endswith('_col')]
        raise ValueError(f"POP UPI file is missing required columns. Could not find: {missing}")

    # --- Aggressively clean the *data* ---
    date_str_cleaned = df[date_col].astype(str).str.replace(r'\s+', '', regex=True)
    norm_df['Std_Date'] = pd.to_datetime(date_str_cleaned, errors='coerce')
    
    amount_str_cleaned = df[amount_col].astype(str).str.replace(r'[\s,]', '', regex=True)
    amount_numeric = pd.to_numeric(amount_str_cleaned, errors='coerce').fillna(0)
    
    type_str_cleaned = df[type_col].astype(str).str.replace(r'\s+', '', regex=True).str.upper()

    is_debit = (type_str_cleaned == 'PAY')
    is_credit = (type_str_cleaned == 'COLLECT')

    norm_df['Std_Debit'] = amount_numeric.where(is_debit, 0)
    norm_df['Std_Credit'] = amount_numeric.where(is_credit, 0)
    
    norm_df['Std_Description'] = np.where(is_debit, 
                                        df[receiver_col].astype(str).str.replace(r'[\n\r]+', ' ', regex=True), 
                                        df[payer_col].astype(str).str.replace(r'[\n\r]+', ' ', regex=True))
    norm_df['Std_Description'] = norm_df['Std_Description'].str.strip()
    
    norm_df['Std_Amount'] = amount_numeric
    norm_df['Std_Type'] = np.where(is_debit, 'DEBIT', 'CREDIT')

    # Final cleanup
    norm_df = norm_df.dropna(subset=['Std_Date']) 
    norm_df = norm_df[(norm_df['Std_Debit'] > 0) | (norm_df['Std_Credit'] > 0)]
    return norm_df


# --- 4. MAIN APP ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part in request.")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.")
    
    if not file.filename.endswith(('.xls', '.xlsx', '.csv')):
        return render_template('index.html', error="Invalid file type. Please upload an Excel (.xls, .xlsx) or CSV (.csv) file.")

    try:
        # --- 1. Read File ---
        all_sheets = {} 
        
        if file.filename.endswith('.csv'):
            try:
                file.stream.seek(0)
                # --- THIS IS THE FIX for the new PhonePe file ---
                # Use header=2 to skip first two rows for PhonePe Statement CSVs
                # We will detect the type *after* reading to decide if header=2 was right
                print("Attempting to read CSV with header=2 (for PhonePe Statement format)...")
                df_sheet = pd.read_csv(file, comment=None, engine='python', header=2)
                # Check if this read actually worked by detecting the type
                file_type_check = detect_file_type(df_sheet.columns)

                if file_type_check != 'PHONEPE_STATEMENT_FORMAT':
                    print("Header=2 likely incorrect, trying header=0 (for POP UPI format)...")
                    # If it's NOT the PhonePe Statement, re-read assuming header is on row 0
                    file.stream.seek(0)
                    df_sheet = pd.read_csv(file, comment=None, engine='python', header=0) 
                    print(f"--- Read CSV (header=0). Columns found: {list(df_sheet.columns)} ---")
                else:
                    print(f"--- Read CSV (header=2). Columns found: {list(df_sheet.columns)} ---")
                
                all_sheets['sheet1'] = df_sheet
                
            except Exception as read_e: # Catch other potential errors
                print(f"General Error reading CSV: {read_e}")
                traceback.print_exc()
                try: # Fallback to latin1 just in case encoding was the issue
                     print("Trying latin1 encoding...")
                     file.stream.seek(0)
                     df_sheet = pd.read_csv(file, comment=None, engine='python', encoding='latin1', header=0) # Assume header=0 for this fallback
                     all_sheets['sheet1'] = df_sheet
                except Exception as fallback_read_e:
                    print(f"Fallback CSV read also failed: {fallback_read_e}")
                    return render_template('index.html', error=f"Error reading CSV file: {read_e}")
        else: # For Excel files
            try:
                file.stream.seek(0) 
                # Excel files usually handle headers better, read normally
                all_sheets = pd.read_excel(file, sheet_name=None) 
                print(f"--- Read Excel. Sheets found: {list(all_sheets.keys())} ---")
            except Exception as read_e:
                print(f"Error reading Excel: {read_e}")
                traceback.print_exc()
                return render_template('index.html', error=f"Error reading Excel file: {read_e}")

        # --- 2. Detect and Process ---
        all_dfs_normalized = []
        for sheet_name, df_sheet in all_sheets.items():
            if df_sheet.empty:
                print(f"Skipping empty sheet '{sheet_name}'.")
                continue
            
            # --- Detect format based on columns read ---
            file_type = detect_file_type(df_sheet.columns) 
            
            try:
                if file_type == 'POP_UPI_FORMAT':
                    norm_df = process_pop_upi_format(df_sheet)
                # === Use the NEW function name ===
                elif file_type == 'PHONEPE_STATEMENT_FORMAT': 
                    norm_df = process_phonepe_statement_format(df_sheet)
                else:
                    print(f"Skipping sheet '{sheet_name}': No processor found for UNKNOWN_FORMAT.")
                    continue 
                
                if norm_df is not None and not norm_df.empty:
                    all_dfs_normalized.append(norm_df)
                else:
                     print(f"Processor for '{file_type}' returned empty DataFrame for sheet '{sheet_name}'.")

            except Exception as process_e:
                print(f"Error processing sheet '{sheet_name}' with detected format '{file_type}': {process_e}")
                traceback.print_exc()
                # Crucially, display the specific error to the user
                return render_template('index.html', error=f"Error processing file '{file.filename}': {process_e}")
        
        if not all_dfs_normalized:
            # If after trying all sheets, none were valid
            return render_template('index.html', error="Could not find any sheets with valid and processable transaction data.")

        # --- 3. Concatenate and Analyze ---
        df = pd.concat(all_dfs_normalized, ignore_index=True)
        df = df.sort_values(by='Std_Date')

        if df.empty:
            # This error should now be much harder to reach
            return render_template('index.html', error="No valid transaction data found after cleaning across all sheets.")

        start_date = df['Std_Date'].min().strftime('%B %d, %Y')
        end_date = df['Std_Date'].max().strftime('%B %d, %Y')

        # --- 4. Perform Analyses (Using Standardized Names - NO CHANGES NEEDED BELOW) ---
        
        # Analysis 1: Payee & Category
        df['Category'] = df['Std_Description'].apply(categorize_transaction)

        # Analysis 2: Top 10 Spending
        top_10_spending = pd.DataFrame()
        # Filter out generic transfer descriptions before grouping
        spending_df = df[(df['Std_Type'] == 'DEBIT') & 
                         (df['Std_Description'] != 'N/A') & 
                         (~df['Std_Description'].str.startswith('Transfer to', na=False))] 
        if not spending_df.empty:
            top_10_spending = spending_df.groupby('Std_Description')['Std_Debit'].sum().nlargest(10).reset_index().rename(columns={'Std_Description': 'PayeeOrDetails', 'Std_Debit': 'Amount'})

        # Analysis 3: Monthly Summary
        monthly_summary = pd.DataFrame()
        df['Month'] = df['Std_Date'].dt.to_period('M')
        monthly_summary = df.groupby('Month')[['Std_Debit', 'Std_Credit']].sum().reset_index()
        monthly_summary = monthly_summary.rename(columns={'Std_Debit': 'Debit', 'Std_Credit': 'Credit'}) 
        if not monthly_summary.empty:
            monthly_summary['MonthStr'] = monthly_summary['Month'].astype(str)

        # Analysis 4: Total Debit vs Credit
        total_debit = df['Std_Debit'].sum()
        total_credit = df['Std_Credit'].sum()

        # Analysis 5: Spending by Day of Week
        day_of_week_spending = pd.Series(dtype=float)
        df['Day of Week'] = df['Std_Date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_of_week_spending = df[df['Std_Type'] == 'DEBIT'].groupby('Day of Week')['Std_Debit'].sum().reindex(day_order).fillna(0)

        # Analysis 6: Spending by Category
        category_spending = pd.Series(dtype=float)
        category_spending = df[df['Std_Type'] == 'DEBIT'].groupby('Category')['Std_Debit'].sum().sort_values(ascending=False)

        # Analysis 7: Largest Transactions
        largest_debits_list = []
        largest_credits_list = []
        if total_debit > 0:
            largest_debits = df[df['Std_Debit'] > 0].nlargest(5, 'Std_Debit')[['Std_Date', 'Std_Description', 'Std_Debit']]
            largest_debits_list = largest_debits.to_dict('records')
            for item in largest_debits_list:
                item['Date'] = item['Std_Date'].strftime('%b %d, %Y')
                item['PayeeOrDetails'] = item['Std_Description']
                item['Debit'] = f"₹{item['Std_Debit']:,.2f}"
        if total_credit > 0:
            largest_credits = df[df['Std_Credit'] > 0].nlargest(5, 'Std_Credit')[['Std_Date', 'Std_Description', 'Std_Credit']]
            largest_credits_list = largest_credits.to_dict('records')
            for item in largest_credits_list:
                item['Date'] = item['Std_Date'].strftime('%b %d, %Y')
                item['PayeeOrDetails'] = item['Std_Description']
                item['Credit'] = f"₹{item['Std_Credit']:,.2f}"


        # --- 5. Prepare Excel Summary ---
        summary_filename = 'summary_report_with_graphs.xlsx'
        summary_path = os.path.join(app.config['UPLOAD_FOLDER'], summary_filename)
        graph_paths = {}

        # --- 6. Generate Graphs ---

        # Graph 1: Spending by Category Pie Chart
        category_pie_filename = 'category_spending_pie.png'
        category_pie_url = None
        category_pie_path = None
        if not category_spending.empty and category_spending.sum() > 0:
            plt.figure(figsize=(10, 8))
            threshold = category_spending.sum() * 0.02
            small_categories = category_spending[category_spending < threshold] if threshold > 0 else pd.Series(dtype=float)
            large_categories = category_spending[category_spending >= threshold]
            plot_data = large_categories.copy()
            if not small_categories.empty:
                other_sum = small_categories.sum()
                if other_sum > 0:
                      plot_data['Other (<2%)'] = other_sum
            plot_data = plot_data.sort_values(ascending=False)
            if not plot_data.empty:
                pie_labels = [f'{cat}\n(₹{amt:,.0f})' for cat, amt in plot_data.items()]
                colors = plt.cm.viridis(range(len(plot_data)))
                plt.pie(plot_data, labels=pie_labels, colors=colors, autopct='%1.1f%%',
                        startangle=90, pctdistance=0.85,
                        textprops={'color': 'white', 'fontsize': 10, 'fontweight':'bold'},
                        wedgeprops={'edgecolor': 'white', 'linewidth': 0.5})
                centre_circle = plt.Circle((0,0),0.70,fc='none')
                fig = plt.gcf(); fig.gca().add_artist(centre_circle)
                plt.title('Spending Distribution by Category', color='white', fontsize=16, pad=20)
                plt.tight_layout()
                category_pie_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], category_pie_filename)
                plt.savefig(category_pie_path, transparent=True, bbox_inches='tight')
                category_pie_url = f'images/{category_pie_filename}'
                if os.path.exists(category_pie_path): graph_paths['category'] = category_pie_path
            plt.close()

        # Graph 2: Monthly Debit/Credit Bar Chart
        graph_filename = 'monthly_summary.png'
        graph_url = None
        graph_path = None
        if not monthly_summary.empty: 
            monthly_summary_melted_bar = monthly_summary.melt('MonthStr', var_name='Type', value_name='Amount', value_vars=['Debit', 'Credit'])
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(data=monthly_summary_melted_bar, x='MonthStr', y='Amount', hue='Type', palette={'Debit': '#E74C3C', 'Credit': '#2ECC71'})
            ax.set_title('Monthly Debit vs. Credit', color='white', fontsize=16, pad=20)
            ax.tick_params(axis='x', colors='white', rotation=45)
            ax.tick_params(axis='y', colors='white')
            ax.set_xlabel('Month', color='white', fontsize=12)
            ax.set_ylabel('Amount (₹)', color='white', fontsize=12)
            ax.legend(title_fontsize='13', labelcolor='white')
            ax.set_facecolor('none')
            for p in ax.patches:
                height = p.get_height()
                if pd.notna(height) and height > 0:
                    ax.annotate(f'₹{height:,.0f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', color='white', fontsize=10)
            plt.tight_layout()
            graph_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], graph_filename)
            plt.savefig(graph_path, transparent=True, bbox_inches='tight')
            graph_url = f'images/{graph_filename}'
            if os.path.exists(graph_path): graph_paths['monthly_bar'] = graph_path
            plt.close()

        # Graph 3: Cumulative Balance
        cum_graph_filename = 'cumulative_balance.png'
        cum_graph_url = None
        cum_graph_path = None
        if not df.empty:
            df['Balance'] = df['Std_Credit'].cumsum() - df['Std_Debit'].cumsum()
            plt.figure(figsize=(10, 6))
            ax = sns.lineplot(data=df, x='Std_Date', y='Balance', color='#00f2ea', linewidth=2.5)
            ax.set_title('Cumulative Balance Over Time', color='white', fontsize=16, pad=20)
            ax.tick_params(axis='x', colors='white', rotation=45)
            ax.tick_params(axis='y', colors='white')
            ax.set_xlabel('Date', color='white', fontsize=12)
            ax.set_ylabel('Balance (₹)', color='white', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.3, color='white')
            ax.set_facecolor('none')
            plt.tight_layout()
            cum_graph_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], cum_graph_filename)
            plt.savefig(cum_graph_path, transparent=True, bbox_inches='tight')
            cum_graph_url = f'images/{cum_graph_filename}'
            if os.path.exists(cum_graph_path): graph_paths['cumulative'] = cum_graph_path
            plt.close()

        # Graph 4: Top 10 Spending
        top_10_graph_filename = 'top_10_spending.png'
        top_10_graph_url = None
        top_10_graph_path = None
        if not top_10_spending.empty:
            plt.figure(figsize=(10, 8))
            ax = sns.barplot(data=top_10_spending, y='PayeeOrDetails', x='Amount', palette='viridis_r')
            ax.set_title('Top 10 Spending (by Payee/Details)', color='white', fontsize=16, pad=20)
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white', labelsize=10)
            ax.set_xlabel('Total Amount (₹)', color='white', fontsize=12)
            ax.set_ylabel('Payee / Details', color='white', fontsize=12)
            ax.set_facecolor('none')
            for i, (value, name) in enumerate(zip(top_10_spending['Amount'], top_10_spending['PayeeOrDetails'])):
                if pd.notna(value):
                    ax.text(value + 1, i, f' ₹{value:,.0f}', va='center', ha='left', color='white', fontsize=10)
            plt.tight_layout()
            top_10_graph_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], top_10_graph_filename)
            plt.savefig(top_10_graph_path, transparent=True, bbox_inches='tight')
            plt.close()
            top_10_graph_url = f'images/{top_10_graph_filename}'
            if os.path.exists(top_10_graph_path): graph_paths['top_10'] = top_10_graph_path

        # Graph 5: Debit vs Credit Pie Chart
        pie_chart_filename = 'debit_credit_pie.png'
        pie_chart_url = None
        pie_chart_path = None
        if total_debit > 0 or total_credit > 0:
            pie_data = [total_debit, total_credit]
            pie_labels = [f'Total Debit\n(₹{total_debit:,.0f})', f'Total Credit\n(₹{total_credit:,.0f})']
            colors = ['#E74C3C', '#2ECC71']
            plt.figure(figsize=(8, 8))
            plt.pie(pie_data, labels=pie_labels, colors=colors, autopct='%1.1f%%', textprops={'color': 'white', 'fontsize': 12, 'fontweight': 'bold'}, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
            plt.title('Total Debit vs. Credit', color='white', fontsize=16, pad=20)
            plt.tight_layout()
            pie_chart_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], pie_chart_filename)
            plt.savefig(pie_chart_path, transparent=True, bbox_inches='tight')
            plt.close()
            pie_chart_url = f'images/{pie_chart_filename}'
            if os.path.exists(pie_chart_path): graph_paths['debit_credit'] = pie_chart_path

        # Graph 6: Spending by Day of Week
        day_of_week_filename = 'day_of_week_spending.png'
        day_of_week_url = None
        day_of_week_path = None
        day_of_week_spending_filled = day_of_week_spending.fillna(0)
        if not day_of_week_spending_filled.eq(0).all():
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=day_of_week_spending_filled.index, y=day_of_week_spending_filled.values, palette='plasma')
            ax.set_title('Spending by Day of Week', color='white', fontsize=16, pad=20)
            ax.tick_params(axis='x', colors='white', rotation=45)
            ax.tick_params(axis='y', colors='white')
            ax.set_xlabel('Day of Week', color='white', fontsize=12)
            ax.set_ylabel('Total Amount (₹)', color='white', fontsize=12)
            ax.set_facecolor('none')
            for p in ax.patches:
                height = p.get_height()
                if pd.notna(height) and height > 0:
                    ax.annotate(f'₹{height:,.0f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', color='white', fontsize=10)
            plt.tight_layout()
            day_of_week_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], day_of_week_filename)
            plt.savefig(day_of_week_path, transparent=True, bbox_inches='tight')
            plt.close()
            day_of_week_url = f'images/{day_of_week_filename}'
            if os.path.exists(day_of_week_path): graph_paths['day_of_week'] = day_of_week_path

        # Graph 7: Income vs. Expense Trend
        trend_filename = 'income_expense_trend.png'
        trend_url = None
        trend_path = None
        if not monthly_summary.empty:
            monthly_summary['MonthDate'] = monthly_summary['Month'].dt.to_timestamp()
            plt.figure(figsize=(10, 6))
            plt.plot(monthly_summary['MonthDate'], monthly_summary['Credit'], marker='o', linestyle='-', color='#2ECC71', linewidth=2.5, label='Income (Credit)')
            plt.plot(monthly_summary['MonthDate'], monthly_summary['Debit'], marker='o', linestyle='-', color='#E74C3C', linewidth=2.5, label='Expenses (Debit)')
            plt.title('Monthly Income vs. Expense Trend', color='white', fontsize=16, pad=20)
            plt.xlabel('Month', color='white', fontsize=12)
            plt.ylabel('Amount (₹)', color='white', fontsize=12)
            plt.xticks(rotation=45, color='white')
            plt.yticks(color='white')
            plt.grid(True, linestyle='--', alpha=0.3, color='white')
            plt.legend(labelcolor='white')
            plt.gca().set_facecolor('none')
            plt.tight_layout()
            trend_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], trend_filename)
            plt.savefig(trend_path, transparent=True, bbox_inches='tight')
            plt.close()
            trend_url = f'images/{trend_filename}'
            if os.path.exists(trend_path): graph_paths['trend'] = trend_path


        # --- 7. Save Excel Summary WITH Images ---
        excel_write_successful = False
        if not monthly_summary.empty:
            try:
                with pd.ExcelWriter(summary_path, engine='xlsxwriter') as writer:
                    summary_df_to_write = monthly_summary[['MonthStr', 'Debit', 'Credit']].rename(columns={'MonthStr':'Month'})
                    summary_df_to_write.to_excel(writer, index=False, sheet_name='Summary', startrow=0)

                    workbook = writer.book
                    worksheet = writer.sheets['Summary']
                    insert_row = len(summary_df_to_write) + 5
                    image_options = {'x_scale': 0.5, 'y_scale': 0.5}

                    image_order = ['category', 'debit_credit', 'trend', 'day_of_week', 'monthly_bar', 'cumulative', 'top_10']
                    row_heights = {'category': 30, 'debit_credit': 30, 'trend': 25, 'day_of_week': 25, 'monthly_bar': 25, 'cumulative': 25, 'top_10': 30}

                    col_num = 0
                    max_col_images = 2
                    current_col_count = 0

                    for key in image_order:
                        if key in graph_paths and graph_paths[key]:
                            cell = f"{chr(ord('A') + col_num * 10)}{insert_row}"
                            worksheet.insert_image(cell, graph_paths[key], image_options)

                            current_col_count += 1
                            if current_col_count >= max_col_images:
                                col_num = 0
                                current_col_count = 0
                                insert_row += row_heights.get(key, 25) + 2
                            else:
                                col_num += 1
                excel_write_successful = True # Mark as successful
            except Exception as excel_error:
                print(f"Error writing Excel file with images: {excel_error}")
                # Fallback: Save Excel without images if embedding fails
                try:
                    with pd.ExcelWriter(summary_path, engine='xlsxwriter') as writer:
                        summary_df_to_write = monthly_summary[['MonthStr', 'Debit', 'Credit']].rename(columns={'MonthStr':'Month'})
                        summary_df_to_write.to_excel(writer, index=False, sheet_name='Summary', startrow=0)
                    print("Saved Excel summary without images as fallback.")
                    excel_write_successful = True # Still successful, just without images
                except Exception as fallback_excel_error:
                    print(f"Fallback Excel write also failed: {fallback_excel_error}")

        download_file_to_pass = summary_filename if excel_write_successful else None


        # --- 8. Render the page with results ---
        return render_template('index.html',
                                category_pie_url=category_pie_url,
                                graph_url=graph_url,
                                cum_graph_url=cum_graph_url,
                                top_10_graph_url=top_10_graph_url,
                                pie_chart_url=pie_chart_url,
                                day_of_week_url=day_of_week_url,
                                trend_url=trend_url,
                                download_filename=download_file_to_pass,
                                start_date=start_date,
                                end_date=end_date,
                                largest_debits=largest_debits_list,
                                largest_credits=largest_credits_list
                                )

    except Exception as e:
        print(f"Error during analysis: {e}")
        traceback.print_exc() # Print full error trace
        return render_template('index.html', error=f"An error occurred: {e}. Please ensure it's a valid transaction file.")


@app.route('/download/<filename>')
def download_file(filename):
    if filename != 'summary_report_with_graphs.xlsx':
        return "Invalid filename", 400
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'],
                                    filename,
                                    as_attachment=True)
    except FileNotFoundError:
        return "Summary file not found.", 404

if __name__ == '__main__':
    app.run(debug=True)