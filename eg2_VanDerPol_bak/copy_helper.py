import pandas as pd

# Load the entire Excel file
excel_file = '/Users/shiqing/Desktop/Lipschitz-System-ID/eg2.xlsx'
# Load all sheets
xls = pd.ExcelFile(excel_file)
# List sheet names
print(xls.sheet_names)

# Load the Excel file
df = pd.read_excel(excel_file, sheet_name='Ours 0.25')
column_data = df['Exp Num'].tolist()
print(column_data)

df = pd.read_excel(excel_file, sheet_name='Ours 0.5')
column_data = df['Exp Num'].tolist()
print(column_data)

df = pd.read_excel(excel_file, sheet_name='Ours 1.0')
column_data = df['Exp Num'].tolist()
print(column_data)

