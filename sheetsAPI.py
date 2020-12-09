import gspread
from oauth2client.service_account import ServiceAccountCredentials

class SheetsAPI:
    def __init__(self, startRow):
        scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)
        client = gspread.authorize(creds)
        self.sheet = client.open("MajorProjectDB").sheet1
        self.n=startRow
        print("[INFO] Sheets API Initialized")

    def insertRecord(self, row):
        print("[INFO] Inserted Row:",row)
        self.sheet.insert_row(row, self.n)
        self.n+=1



