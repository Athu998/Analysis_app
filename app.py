import matplotlib
matplotlib.use('Agg') # Use non-GUI backend
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template, send_from_directory, url_for
import io
import re
import pdfplumber # <-- NEW: Import pdfplumber

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
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', text) for keyword in keywords):
            return category
    return 'Other'

# === NEW: PDF Extraction Function ===
def extract_data_from_pdf(pdf_file_stream):
    """
    Attempts to extract transaction data from a PDF file stream using pdfplumber.
    Assumes a table structure similar to PhonePe statements.
    Returns a pandas DataFrame or None if extraction fails.
    """
    all_transactions = []
    expected_header = ['Date', 'Transaction Details', 'Type', 'Amount'] # Adjust if needed

    try:
        with pdfplumber.open(pdf_file_stream) as pdf:
            for page in pdf.pages:
                # Try extracting tables first
                tables = page.extract_tables()
                found_table = False
                for table in tables:
                    if table and len(table) > 1: # Check if table exists and has rows
                        header = [str(h).strip() if h is not None else '' for h in table[0]] # Get header
                        # Check if header looks like our expected transaction table header
                        # This is a basic check, might need adjustment
                        if all(col in header for col in ['Date', 'Type', 'Amount']):
                            found_table = True
                            # Find indices of required columns
                            try:
                                date_idx = header.index('Date')
                                details_idx = header.index('Transaction Details')
                                type_idx = header.index('Type')
                                amount_idx = header.index('Amount')
                            except ValueError:
                                print(f"Warning: Could not find all expected columns in table header: {header}")
                                continue # Skip this table if essential columns missing

                            for row in table[1:]: # Skip header row
                                # Ensure row has enough columns
                                if len(row) > max(date_idx, details_idx, type_idx, amount_idx):
                                    date = str(row[date_idx]).strip() if row[date_idx] else None
                                    details = str(row[details_idx]).strip() if row[details_idx] else ''
                                    txn_type = str(row[type_idx]).strip().upper() if row[type_idx] else None
                                    amount_str = str(row[amount_idx]).strip() if row[amount_idx] else None

                                    # Basic validation: need date, type, and amount
                                    if date and txn_type in ['DEBIT', 'CREDIT'] and amount_str:
                                         # Attempt to clean amount (remove currency symbols, commas)
                                        try:
                                            amount = float(re.sub(r'[₹,]', '', amount_str))
                                            all_transactions.append({
                                                'Date': date,
                                                'Transaction Details': details,
                                                'Type': txn_type,
                                                'Amount': amount
                                            })
                                        except (ValueError, TypeError):
                                            print(f"Warning: Could not parse amount '{amount_str}' in row: {row}")
                                            continue # Skip row if amount is invalid
                                else:
                                     # Sometimes extra info is extracted as rows, try to skip them
                                     # print(f"Skipping row with potentially missing data: {row}")
                                     pass


                # Basic Text extraction fallback (less reliable) - You might need to add this
                # if not found_table:
                #     text = page.extract_text()
                #     # Add complex regex logic here to parse the text line by line
                #     # This is highly dependent on the PDF's specific text layout
                #     pass

        if not all_transactions:
            return None # No transactions found

        df = pd.DataFrame(all_transactions)
        # Convert Date - might need more robust parsing depending on PDF format
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'Amount']) # Ensure essential data is valid
        return df

    except Exception as e:
        print(f"Error reading PDF: {e}")
        import traceback
        traceback.print_exc()
        return None
# === END NEW PDF Function ===


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

    # === UPDATED: Check for PDF extension ===
    if not file.filename.lower().endswith('.pdf'):
        return render_template('index.html', error="Invalid file type. Please upload a PDF file (.pdf).")

    try:
        # --- 1. Read & Clean Data (Now from PDF) ---
        df = extract_data_from_pdf(file.stream) # Pass the file stream

        if df is None or df.empty:
            return render_template('index.html', error="Could not extract valid transaction data from the PDF. The format might be unsupported.")

        # --- Data cleaning and analysis proceeds as before ---
        df['Debit'] = df.apply(lambda r: r['Amount'] if r['Type'] == 'DEBIT' else 0, axis=1)
        df['Credit'] = df.apply(lambda r: r['Amount'] if r['Type'] == 'CREDIT' else 0, axis=1)

        start_date = df['Date'].min().strftime('%B %d, %Y')
        end_date = df['Date'].max().strftime('%B %d, %Y')

        # Analysis 1: Payee & Category
        def get_payee(row): # Simplified get_payee for PDF
             details = str(row['Transaction Details'])
             if details.startswith('Paid to '):
                 return details.replace('Paid to ', '').strip()
             # Basic fallback if details exist
             elif details:
                  # Take first line or a limited number of characters? Be careful.
                  return details.split('\n')[0].strip() # Example: Take only first line
             return 'N/A'

        df['PayeeOrDetails'] = df.apply(get_payee, axis=1)
        df['Category'] = df.apply(lambda row: categorize_transaction(row['PayeeOrDetails']), axis=1)

        # Analysis 2: Top 10 Spending
        spending_df = df[(df['Type'] == 'DEBIT') & (df['PayeeOrDetails'] != 'N/A')]
        top_10_spending = spending_df.groupby('PayeeOrDetails')['Amount'].sum().nlargest(10).reset_index()

        # Analysis 3: Monthly Summary
        df['Month'] = df['Date'].dt.to_period('M')
        monthly_summary = df.groupby('Month')[['Debit', 'Credit']].sum().reset_index()
        monthly_summary['MonthStr'] = monthly_summary['Month'].astype(str)

        # Analysis 4: Total Debit vs Credit
        total_debit = df['Debit'].sum()
        total_credit = df['Credit'].sum()

        # Analysis 5: Spending by Day of Week
        df['Day of Week'] = df['Date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_of_week_spending = df[df['Type'] == 'DEBIT'].groupby('Day of Week')['Amount'].sum().reindex(day_order)

        # Analysis 6: Spending by Category
        category_spending = df[df['Type'] == 'DEBIT'].groupby('Category')['Amount'].sum().sort_values(ascending=False)

        # Analysis 7: Largest Transactions
        largest_debits = df[df['Debit'] > 0].nlargest(5, 'Debit')[['Date', 'PayeeOrDetails', 'Debit']]
        largest_credits = df[df['Credit'] > 0].nlargest(5, 'Credit')[['Date', 'PayeeOrDetails', 'Credit']]
        largest_debits_list = largest_debits.to_dict('records')
        largest_credits_list = largest_credits.to_dict('records')
        for item in largest_debits_list:
            item['Date'] = item['Date'].strftime('%b %d, %Y')
            item['Debit'] = f"₹{item['Debit']:,.2f}"
        for item in largest_credits_list:
            item['Date'] = item['Date'].strftime('%b %d, %Y')
            item['Credit'] = f"₹{item['Credit']:,.2f}"

        # --- 3. Save Excel Summary ---
        summary_filename = 'summary_report.xlsx' # Keep Excel download for summary
        summary_path = os.path.join(app.config['UPLOAD_FOLDER'], summary_filename)
        with pd.ExcelWriter(summary_path, engine='xlsxwriter') as writer:
            summary_df_to_write = monthly_summary[['MonthStr', 'Debit', 'Credit']].rename(columns={'MonthStr':'Month'})
            summary_df_to_write.to_excel(writer, index=False, sheet_name='Summary')
            # Note: Embedding images in Excel won't work easily with PDF input unless graphs are regenerated
            # Keeping the simple Excel export for now.

        # --- 4. Generate Graphs (Code remains the same, uses the df from PDF) ---
        graph_paths = {} # Dictionary to store paths of generated graphs

        # Graph 1: Spending by Category Pie Chart
        category_pie_filename = 'category_spending_pie.png'
        category_pie_url = None
        category_pie_path = None
        if not category_spending.empty:
            plt.figure(figsize=(10, 8))
            threshold = category_spending.sum() * 0.02
            small_categories = category_spending[category_spending < threshold]
            large_categories = category_spending[category_spending >= threshold]
            plot_data = large_categories.copy()
            if not small_categories.empty:
                other_sum = small_categories.sum()
                if other_sum > 0:
                     plot_data['Other (<2%)'] = other_sum
            plot_data = plot_data.sort_values(ascending=False)
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
            plt.close()
            category_pie_url = f'images/{category_pie_filename}'
            if os.path.exists(category_pie_path): graph_paths['category'] = category_pie_path

        # Graph 2: Monthly Debit/Credit Bar Chart
        monthly_summary_melted_bar = monthly_summary.melt('MonthStr', var_name='Type', value_name='Amount', value_vars=['Debit', 'Credit'])
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=monthly_summary_melted_bar, x='MonthStr', y='Amount', hue='Type', palette={'Debit': '#E74C3C', 'Credit': '#2ECC71'})
        # ... (rest of graph 2 plotting code remains the same) ...
        plt.tight_layout()
        graph_filename = 'monthly_summary.png'
        graph_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], graph_filename)
        plt.savefig(graph_path, transparent=True, bbox_inches='tight')
        plt.close()
        graph_url = f'images/{graph_filename}'
        if os.path.exists(graph_path): graph_paths['monthly_bar'] = graph_path

        # Graph 3: Cumulative Balance
        df_sorted = df.sort_values(by='Date')
        df_sorted['Balance'] = df_sorted['Credit'].cumsum() - df_sorted['Debit'].cumsum()
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(data=df_sorted, x='Date', y='Balance', color='#00f2ea', linewidth=2.5)
        # ... (rest of graph 3 plotting code remains the same) ...
        plt.tight_layout()
        cum_graph_filename = 'cumulative_balance.png'
        cum_graph_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], cum_graph_filename)
        plt.savefig(cum_graph_path, transparent=True, bbox_inches='tight')
        plt.close()
        cum_graph_url = f'images/{cum_graph_filename}'
        if os.path.exists(cum_graph_path): graph_paths['cumulative'] = cum_graph_path

        # Graph 4: Top 10 Spending
        top_10_graph_filename = 'top_10_spending.png'
        top_10_graph_url = None
        top_10_graph_path = None
        if not top_10_spending.empty:
            plt.figure(figsize=(10, 8))
            ax = sns.barplot(data=top_10_spending, y='PayeeOrDetails', x='Amount', palette='viridis_r')
            # ... (rest of graph 4 plotting code remains the same) ...
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
            # ... (rest of graph 5 plotting code remains the same) ...
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
        if day_of_week_spending is not None and not day_of_week_spending.fillna(0).eq(0).all():
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=day_of_week_spending.index, y=day_of_week_spending.values, palette='plasma')
            # ... (rest of graph 6 plotting code remains the same) ...
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
            # ... (rest of graph 7 plotting code remains the same) ...
            plt.tight_layout()
            trend_path = os.path.join(app.config['STATIC_IMAGES_FOLDER'], trend_filename)
            plt.savefig(trend_path, transparent=True, bbox_inches='tight')
            plt.close()
            trend_url = f'images/{trend_filename}'
            if os.path.exists(trend_path): graph_paths['trend'] = trend_path


        # --- 5. Render the page with results ---
        return render_template('index.html',
                               category_pie_url=category_pie_url,
                               graph_url=graph_url,
                               cum_graph_url=cum_graph_url,
                               top_10_graph_url=top_10_graph_url,
                               pie_chart_url=pie_chart_url,
                               day_of_week_url=day_of_week_url,
                               trend_url=trend_url,
                               download_filename=summary_filename, # Keep Excel download
                               start_date=start_date,
                               end_date=end_date,
                               largest_debits=largest_debits_list,
                               largest_credits=largest_credits_list
                              )

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=f"An error occurred during PDF processing or analysis. Please check file format.")


@app.route('/download/<filename>')
def download_file(filename):
    """Provides the download link for the processed file."""
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename,
                               as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)