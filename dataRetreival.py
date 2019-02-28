

'Charges: 10% of total cost of buy or sell (remember to calculate both ways). Min of $15 each. (for SG markets)'
'NYSE, Japan Exchange, Shanghai Exchange??'

import requests


tiingoApiKey = 'a3bca86e84405419d293f5471c4ae88494407b00'

def main():
    url = 'https://www.investopedia.com/markets/api/partial/historical/?Symbol=[AAPL]&Type=Historical+Prices&Timeframe=Daily&StartDate=Nov+28%2C+2018&EndDate=Dec+05%2C+2018'

    r = requests.get(url)

    print(r.status_code, r.reason)
    print(r.text)


if __name__ == '__main__':
    main()