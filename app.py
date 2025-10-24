import matplotlib
matplotlib.use('Agg') # Use non-GUI backend
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template, send_from_directory, url_for
import io
import re # Import regex for categorization

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
    if not file.filename.endswith(('.xls', '.xlsx')):
        return render_template('index.html', error="Invalid file type. Please upload an Excel file (.xls or .xlsx).")

    try:
        # --- 1. Read & Clean Data ---
        all_sheets = pd.read_excel(file, sheet_name=None)
        transaction_dfs = []
        required_cols = {'Date', 'Transaction Details', 'Type', 'Amount'}
        for sheet_name, df_sheet in all_sheets.items():
            if required_cols.issubset(df_sheet.columns):
                transaction_dfs.append(df_sheet)
        if not transaction_dfs:
            return render_template('index.html', error="Could not find transaction data. Ensure file has 'Date', 'Type', and 'Amount' columns.")
        df = pd.concat(transaction_dfs, ignore_index=True)
        df = df.dropna(subset=['Date', 'Type', 'Amount'])
        # Try specific format first, then let pandas infer if it fails
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y', errors='raise')
        except (ValueError, TypeError):
             df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # Fallback

        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Debit'] = df.apply(lambda r: r['Amount'] if r['Type'] == 'DEBIT' else 0, axis=1)
        df['Credit'] = df.apply(lambda r: r['Amount'] if r['Type'] == 'CREDIT' else 0, axis=1)
        df = df.dropna(subset=['Date', 'Amount'])
        if df.empty:
            return render_template('index.html', error="No valid transaction data found after cleaning.")

        start_date = df['Date'].min().strftime('%B %d, %Y')
        end_date = df['Date'].max().strftime('%B %d, %Y')

        # --- 2. Perform Analyses ---

        # Analysis 1: Payee & Category
        def get_payee(row):
            details = str(row['Transaction Details'])
            if row['Type'] == 'DEBIT' and details.startswith('Paid to '):
                return details.replace('Paid to ', '').strip()
            elif details: # Try getting text from details for Credits or non-"Paid to" Debits
                 # Take first line if details contain newline, otherwise whole detail
                 return details.split('\n')[0].strip()
            return 'N/A' # Use N/A if no suitable text found

        df['PayeeOrDetails'] = df.apply(get_payee, axis=1)
        df['Category'] = df.apply(lambda row: categorize_transaction(row['PayeeOrDetails']), axis=1)

        # Analysis 2: Top 10 Spending (Use PayeeOrDetails for consistency)
        spending_df = df[(df['Type'] == 'DEBIT') & (df['PayeeOrDetails'] != 'N/A')]
        top_10_spending = spending_df.groupby('PayeeOrDetails')['Amount'].sum().nlargest(10).reset_index()

        # Analysis 3: Monthly Summary (Used for multiple things)
        df['Month'] = df['Date'].dt.to_period('M')
        monthly_summary = df.groupby('Month')[['Debit', 'Credit']].sum().reset_index()
        monthly_summary['MonthStr'] = monthly_summary['Month'].astype(str) # String version for plotting/saving

        # Analysis 4: Total Debit vs Credit
        total_debit = df['Debit'].sum()
        total_credit = df['Credit'].sum()

        # Analysis 5: Spending by Day of Week
        df['Day of Week'] = df['Date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # Use fillna(0) in case some days have no spending
        day_of_week_spending = df[df['Type'] == 'DEBIT'].groupby('Day of Week')['Amount'].sum().reindex(day_order).fillna(0)

        # Analysis 6: Spending by Category
        category_spending = df[df['Type'] == 'DEBIT'].groupby('Category')['Amount'].sum().sort_values(ascending=False)

        # === Analysis 7: Largest Transactions ===
        largest_debits = df[df['Debit'] > 0].nlargest(5, 'Debit')[['Date', 'PayeeOrDetails', 'Debit']]
        largest_credits = df[df['Credit'] > 0].nlargest(5, 'Credit')[['Date', 'PayeeOrDetails', 'Credit']]
        # Convert to list of dicts for easier handling in template
        largest_debits_list = largest_debits.to_dict('records')
        largest_credits_list = largest_credits.to_dict('records')
        # Format date and amount for display
        for item in largest_debits_list:
            item['Date'] = item['Date'].strftime('%b %d, %Y')
            item['Debit'] = f"₹{item['Debit']:,.2f}" # Format as currency
        for item in largest_credits_list:
            item['Date'] = item['Date'].strftime('%b %d, %Y')
            item['Credit'] = f"₹{item['Credit']:,.2f}" # Format as currency
        # === END NEW ===

        # --- 3. Prepare Excel Summary (Save later with graphs) ---
        summary_filename = 'summary_report_with_graphs.xlsx'
        summary_path = os.path.join(app.config['UPLOAD_FOLDER'], summary_filename)
        graph_paths = {} # Initialize graph_paths here before generating graphs

        # --- 4. Generate Graphs (Save paths needed for Excel) ---

        # Graph 1: Spending by Category Pie Chart
        category_pie_filename = 'category_spending_pie.png'
        category_pie_url = None
        category_pie_path = None
        if not category_spending.empty and category_spending.sum() > 0: # Check if there's spending
            plt.figure(figsize=(10, 8))
            threshold = category_spending.sum() * 0.02
            # Ensure calculations handle potential zero threshold or empty small_categories
            small_categories = category_spending[category_spending < threshold] if threshold > 0 else pd.Series(dtype=float)
            large_categories = category_spending[category_spending >= threshold]
            plot_data = large_categories.copy()
            if not small_categories.empty:
                other_sum = small_categories.sum()
                if other_sum > 0:
                     plot_data['Other (<2%)'] = other_sum
            plot_data = plot_data.sort_values(ascending=False)
            # Ensure plot_data is not empty before plotting
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
            plt.close() # Close figure even if plotting didn't happen

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
                if pd.notna(height) and height > 0: # Check for NaN height
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
            df_sorted = df.sort_values(by='Date')
            df_sorted['Balance'] = df_sorted['Credit'].cumsum() - df_sorted['Debit'].cumsum()
            plt.figure(figsize=(10, 6))
            ax = sns.lineplot(data=df_sorted, x='Date', y='Balance', color='#00f2ea', linewidth=2.5)
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
                 if pd.notna(value): # Check if value is not NaN
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
        # Check if day_of_week_spending exists and contains non-zero data after fillna
        day_of_week_spending_filled = day_of_week_spending.fillna(0)
        if day_of_week_spending_filled is not None and not day_of_week_spending_filled.eq(0).all():
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

        # === Graph 7: Income vs. Expense Trend ===
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
        # === END NEW ===

        # --- 5. Save Excel Summary WITH Images ---
        with pd.ExcelWriter(summary_path, engine='xlsxwriter') as writer:
            summary_df_to_write = monthly_summary[['MonthStr', 'Debit', 'Credit']].rename(columns={'MonthStr':'Month'})
            summary_df_to_write.to_excel(writer, index=False, sheet_name='Summary', startrow=0) # Start data at row 0

            workbook = writer.book
            worksheet = writer.sheets['Summary']
            # Start images further down to avoid overlap if data is long
            insert_row = len(summary_df_to_write) + 5
            image_options = {'x_scale': 0.5, 'y_scale': 0.5}

            image_order = ['category', 'debit_credit', 'trend', 'day_of_week', 'monthly_bar', 'cumulative', 'top_10']
            row_heights = {'category': 30, 'debit_credit': 30, 'trend': 25, 'day_of_week': 25, 'monthly_bar': 25, 'cumulative': 25, 'top_10': 30}

            col_num = 0 # Start inserting images in the first column (A)
            max_col_images = 2 # How many images side-by-side
            current_col_count = 0

            for key in image_order:
                 if key in graph_paths:
                     # Calculate cell based on current column and row
                     cell = f"{chr(ord('A') + col_num * 10)}{insert_row}" # Approx 10 columns width per image
                     worksheet.insert_image(cell, graph_paths[key], image_options)

                     current_col_count += 1
                     if current_col_count >= max_col_images:
                         # Move to next row after placing max images per row
                         col_num = 0
                         current_col_count = 0
                         insert_row += row_heights.get(key, 25) + 2 # Add padding rows
                     else:
                         # Move to next column position for the next image
                         col_num += 1
                 # If the loop ends and we didn't fill the last row, move insert_row down anyway
                 # This might not be strictly necessary depending on exact layout needs
                 # if key == image_order[-1] and current_col_count > 0:
                 #    insert_row += row_heights.get(key, 25) + 2


        # --- 6. Render the page with results ---
        return render_template('index.html',
                               category_pie_url=category_pie_url,
                               graph_url=graph_url,
                               cum_graph_url=cum_graph_url,
                               top_10_graph_url=top_10_graph_url,
                               pie_chart_url=pie_chart_url,
                               day_of_week_url=day_of_week_url,
                               trend_url=trend_url,
                               download_filename=summary_filename,
                               start_date=start_date,
                               end_date=end_date,
                               largest_debits=largest_debits_list,
                               largest_credits=largest_credits_list
                              )

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        # Provide a user-friendly error message
        return render_template('index.html', error=f"An error occurred processing the file. Please ensure it's a valid Excel statement and try again.")


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename,
                               as_attachment=True)

# Remove the redundant run block
# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    app.run(debug=True) # Keep this for local development