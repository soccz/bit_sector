import requests
import time
import datetime
import pandas as pd
import yfinance as yf
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

### 1. 업비트 원화 코인 데이터 수집
def get_upbit_krw_data():
    market_url = "https://api.upbit.com/v1/market/all"
    headers = {"Accept": "application/json"}
    
    response = requests.get(market_url, headers=headers)
    markets = response.json()

    # 원화(KRW) 마켓만 필터링
    krw_markets = [market['market'] for market in markets if market['market'].startswith('KRW')]

    # 시작 및 종료 날짜 설정 (3년간 데이터)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365 * 3)

    # 코인 데이터를 수집하는 함수
    def get_daily_data(market, start_date, end_date):
        candle_url = "https://api.upbit.com/v1/candles/days"
        data_list = []
        temp_end_date = end_date

        while temp_end_date > start_date:
            params = {
                'market': market,
                'count': 200,
                'to': temp_end_date.strftime('%Y-%m-%dT%H:%M:%S')
            }

            response = requests.get(candle_url, headers=headers, params=params)
            data = response.json()

            # 빈 응답일 경우 반복을 종료하지 않고 계속 가져옴
            if not data:
                print(f"No data found for {market} from {temp_end_date.strftime('%Y-%m-%d')}")
                break

            data_list.extend(data)

            # data가 비어있지 않을 경우에만 temp_end_date를 갱신
            if data and len(data) > 0:   # 데이터가 있는 경우에만 temp_end_date 갱신
                temp_end_date = datetime.datetime.strptime(data[-1]['candle_date_time_utc'], '%Y-%m-%dT%H:%M:%S') - datetime.timedelta(days=1)
            else:
                break

        # 데이터가 있으면 DataFrame으로 변환
        if data_list:
            return pd.DataFrame(data_list)
        else:
            return pd.DataFrame()  # 빈 DataFrame 반환

    coin_data = {}
    for market in krw_markets:
        print(f"Fetching data for {market}")
        market_data = get_daily_data(market, start_date, end_date)
        if not market_data.empty:
            coin_data[market] = market_data  # 빈 데이터가 아닌 경우에만 저장
        else:
            print(f"No data for {market}, skipping.")

    return coin_data

# 업비트 원화 코인 데이터 가져오기
coin_data = get_upbit_krw_data()

### 2. 주식 데이터 수집 및 섹터별 평균 계산
sectors = {
    "Technology": ["MSFT", "AAPL", "NVDA", "TSM", "AVGO", "ADI", "AMAT", "AMD", "CSCO", "CRM", "GFS", "INTC", "LRCX", "MCHP", "MRVL", "MU", "NXPI", "ORCL", "QCOM", "SNPS", "TER", "TXN", "WDC", "ZBRA", "KLAC", "CDNS", "ANSS", "AKAM", "FTNT"],
    "Financial Services": ["BRK-A", "JPM", "MA", "BAC", "WFC", "C", "GS", "MS", "SCHW", "AXP", "BK", "BLK", "TROW", "STT", "PNC", "USB", "CME", "ICE", "MMC", "AON", "MET", "PRU", "AFL", "LNC", "CINF", "ALL", "PGR", "TRV", "CB", "HIG"],
    "Communication Services": ["GOOGL", "GOOG", "NFLX", "TMUS", "DIS", "CMCSA", "CHTR", "T", "VZ", "FOXA", "FOX", "IRDM", "SIRI", "AMX", "TKC", "TU", "BCE", "TEF", "ORAN", "DTEGY", "CHT", "SKM", "KT", "LUMN", "LILA", "LILAK", "CCOI", "WBD"],
    "Healthcare": ["LLY", "NVO", "UNH", "JNJ", "MRK", "PFE", "ABBV", "ABT", "TMO", "MDT", "BMY", "AMGN", "GILD", "DHR", "ISRG", "REGN", "BIIB", "MRNA", "DXCM", "IDXX", "SYK", "EW", "BSX", "ZBH", "CNC", "RMD", "STE", "MASI", "ALGN", "HOLX"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "TM", "MCD", "NKE", "SBUX", "LOW", "TGT", "TJX", "EBAY", "ROST", "MAR", "YUM", "DG", "CMG", "ULTA", "AZO", "BBY", "DHI", "LEN", "PZZA", "W", "RH", "TPX", "DKNG", "LVS", "WYNN", "BKNG", "ORLY"],
    "Industrials": ["CAT", "GE", "UNP", "RTX", "ETN", "HON", "UPS", "BA", "MMM", "DE", "LMT", "NSC", "CSX", "DAL", "AAL", "UAL", "LUV", "EXPD", "CHRW", "R", "JBHT", "KNX", "WERN", "ODFL", "HUBG", "SNDR", "MRTN", "HTLD", "ITW", "GD"],
    "Consumer Defensive": ["WMT", "PG", "COST", "KO", "PEP", "PM", "MO", "KHC", "MDLZ", "STZ", "KDP", "GIS", "CL", "CPB", "HSY", "K", "TSN", "SJM", "MKC", "CHD", "LW", "HRL", "CAG", "FDP", "HAIN", "THS", "SPTN", "BGS", "FARM", "FLO"],
    "Energy": ["XOM", "CVX", "TTE", "COP", "SLB", "OXY", "PSX", "EOG", "MPC", "HAL", "MRO", "DVN", "EPD", "WMB", "KMI", "TRGP", "OKE", "ET", "LNG", "ENLC", "HESM", "BSM", "AM"],
    "Basic Materials": ["LIN", "BHP", "RIO", "SCCO", "SHW", "APD", "NEM", "DD", "DOW", "PPG", "ECL", "FMC", "ALB", "CE", "EMN", "AVNT", "AXTA", "ASH", "NEU", "IOSP", "OLN", "WLK", "HUN", "CC", "CBT", "TROX"],
    "Real Estate": ["PLD", "AMT", "EQIX", "WELL", "SPG", "PSA", "DLR", "VTR", "AVB", "ESS", "EQR", "UDR", "MAA", "SUI", "CPT", "HST", "AIRC", "ARE", "BXP", "SLG", "FRT", "REG", "KIM", "BRX", "ROIC", "MAC", "AKR"],
    "Utilities": ["NEE", "DUK", "SO", "EXC", "AEP", "XEL", "ES", "PEG", "EIX", "D", "WEC", "PCG", "SRE", "ED", "FE", "PPL", "AEE", "CMS", "DTE", "LNT", "NI", "AWK", "CNP", "NRG", "ETR", "PNW", "VST", "OGE", "IDA"]
}

# 주식 섹터별 평균 데이터를 수집하는 함수
def get_sector_avg_data(sectors):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365 * 3)
    
    sector_avg_data = {}

    for sector, stocks in sectors.items():
        sector_data = []
        for stock in stocks:
            print(f"Fetching data for {stock} in {sector}")
            stock_data = yf.download(stock, start=start_date, end=end_date)
            if not stock_data.empty:
                sector_data.append(stock_data['Close'])

        if sector_data:
            sector_df = pd.concat(sector_data, axis=1)
            sector_avg_data[sector] = sector_df.mean(axis=1)  # 섹터별 평균 계산

    return sector_avg_data

# 주식 섹터별 데이터 가져오기
sector_avg_data = get_sector_avg_data(sectors)

### 3. 모든 원화 코인에 대해 날짜 동기화 및 결측치 처리
def align_and_impute(coin_dates, sector_data):
    # 코인 날짜에 맞게 주식 데이터를 정렬
    sector_data = sector_data.reindex(coin_dates)

    # 데이터가 1차원 시리즈일 경우 2차원 데이터프레임으로 변환
    if isinstance(sector_data, pd.Series):
        sector_data = sector_data.to_frame()

    # KNN Imputer를 사용해 결측치 채움
    imputer = KNNImputer(n_neighbors=5)
    sector_data_imputed = imputer.fit_transform(sector_data)
    
    # 2차원 데이터프레임으로 반환
    return pd.DataFrame(sector_data_imputed, index=coin_dates, columns=sector_data.columns)

# 모든 코인 데이터에 대해 날짜를 동기화하고 결측치 처리
def process_all_coins(coin_data):
    all_coin_data = {}
    coin_dates = pd.to_datetime(coin_data['KRW-BTC']['candle_date_time_kst']).dt.date  # 비트코인 날짜 기준으로 처리

    for market, data in coin_data.items():
        close_prices = data.set_index(pd.to_datetime(data['candle_date_time_kst']).dt.date)['trade_price']
        all_coin_data[market] = close_prices.reindex(coin_dates)

    return pd.DataFrame(all_coin_data)

# 모든 원화 코인 데이터 처리
processed_coin_data = process_all_coins(coin_data)

# 섹터별로 날짜 맞춤 및 결측치 처리
imputed_sector_data = {}
for sector, data in sector_avg_data.items():
    aligned_data = align_and_impute(processed_coin_data.index, data)
    imputed_sector_data[sector] = aligned_data

### 4. 모든 데이터를 하나의 데이터프레임으로 결합하고 CSV로 저장
def combine_and_save_data(processed_coin_data, imputed_sector_data):
    # 섹터 평균 데이터를 하나의 데이터프레임으로 결합
    combined_sector_data = pd.DataFrame(index=processed_coin_data.index)
    for sector, data in imputed_sector_data.items():
        combined_sector_data[sector] = data.values.flatten()

    # 코인 데이터와 섹터 평균 데이터를 결합
    final_combined_data = pd.concat([processed_coin_data, combined_sector_data], axis=1)

    # CSV 파일로 저장
    final_combined_data.to_csv('combined_crypto_sector_data.csv', index=True)
    print("데이터가 combined_crypto_sector_data.csv로 저장되었습니다.")

# 모든 데이터 결합 및 저장 실행
combine_and_save_data(processed_coin_data, imputed_sector_data)

### 5. 상관관계 및 주성분 분석 (PCA) 및 시각화 결과 저장
def perform_correlation_and_pca(coin_data, imputed_sector_data, save_image_path='pca_correlation_visualization.png'):
    # 새로운 Figure 생성
    plt.figure(figsize=(12, 8))
    
    plot_index = 1  # 서브플롯 인덱스
    for coin, coin_close in coin_data.items():
        btc_shifted = coin_close.shift(-3)  # 각 코인 데이터를 3일 전으로 시프트

        for sector, sector_data in imputed_sector_data.items():
            # 상관관계 분석
            combined_data = pd.concat([btc_shifted, sector_data], axis=1, join='inner')
            combined_data.columns = [f'{coin}_Close', f'{sector}_Avg']

            # 결측치 제거 후 데이터가 충분한지 확인
            combined_for_pca = combined_data.dropna()
            if combined_for_pca.shape[0] < 2:
                print(f"데이터가 부족하여 {coin}과 {sector} 섹터에 대한 PCA를 건너뜁니다.")
                continue  # 데이터가 부족할 경우 건너뛰기

            # PCA 분석
            correlation = combined_data.corr().iloc[0, 1]
            print(f"3일 전 {coin}과 {sector} 섹터의 상관관계: {correlation}")
            
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(combined_for_pca)
            
            # PCA 시각화 (서브플롯)
            plt.subplot(len(coin_data), len(imputed_sector_data), plot_index)
            plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue')
            plt.title(f'{coin} vs {sector}')
            plt.xlabel('PC 1')
            plt.ylabel('PC 2')

            plot_index += 1

    # 모든 시각화 결과를 이미지 파일로 저장
    plt.tight_layout()  # 서브플롯 간의 간격 조정
    plt.savefig(save_image_path, format='png')
    print(f"시각화 결과가 {save_image_path}로 저장되었습니다.")
    plt.show()

# 상관관계 및 PCA 분석 실행 및 이미지 저장
perform_correlation_and_pca(processed_coin_data, imputed_sector_data)
