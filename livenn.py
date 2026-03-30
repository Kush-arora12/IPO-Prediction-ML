import pandas as pd
import numpy as np
import time
import re
import yfinance as yf
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


def get_nifty_features():
    print("Fetching Nifty 50 data...")
    nifty = yf.download("^NSEI", period="10y", interval="1d", auto_adjust=True)
    nifty = nifty[['Close']].copy()
    nifty.columns = ['Nifty_Close']
    nifty.index = pd.to_datetime(nifty.index)

    nifty['Nifty_Ret_5d']  = nifty['Nifty_Close'].pct_change(5)  * 100
    nifty['Nifty_Ret_20d'] = nifty['Nifty_Close'].pct_change(20) * 100
    nifty['Nifty_Ret_50d'] = nifty['Nifty_Close'].pct_change(50) * 100

    daily_ret = nifty['Nifty_Close'].pct_change()
    nifty['Nifty_Vol_20d'] = daily_ret.rolling(20).std() * 100

    nifty['Nifty_MA50']  = nifty['Nifty_Close'].rolling(50).mean()
    nifty['Nifty_MA200'] = nifty['Nifty_Close'].rolling(200).mean()
    nifty['Above_MA50']  = (nifty['Nifty_Close'] > nifty['Nifty_MA50']).astype(int)
    nifty['Above_MA200'] = (nifty['Nifty_Close'] > nifty['Nifty_MA200']).astype(int)

    delta = nifty['Nifty_Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    nifty['Nifty_RSI'] = 100 - (100 / (1 + rs))

    nifty.dropna(inplace=True)

    nifty_feature_cols = [
        'Nifty_Ret_5d', 'Nifty_Ret_20d', 'Nifty_Ret_50d',
        'Nifty_Vol_20d', 'Above_MA50', 'Above_MA200', 'Nifty_RSI'
    ]

    print(f"Nifty data loaded: {len(nifty)} trading days ✅")
    return nifty, nifty_feature_cols


def train_ipo_model(nifty, nifty_feature_cols):
    print("Training Neural Network model...")
    try:
        df = pd.read_excel("ipo_data.xlsx")
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'listing' in col.lower():
                date_col = col
                break

        if date_col is None:
            raise ValueError("No date column found in ipo_data.xlsx.")

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce').astype('datetime64[ns]')
        df = df.dropna(subset=[date_col])

        nifty_reset = nifty[nifty_feature_cols].reset_index()
        nifty_reset.columns = ['Date'] + nifty_feature_cols
        nifty_reset['Date'] = pd.to_datetime(nifty_reset['Date']).astype('datetime64[ns]')

        df = pd.merge_asof(
            df.sort_values(date_col),
            nifty_reset.sort_values('Date'),
            left_on=date_col,
            right_on='Date',
            direction='backward'
        )
        df = df.dropna(subset=nifty_feature_cols)

        ipo_features = ['Issue_Size(crores)', 'QIB', 'HNI', 'RII', 'Total', 'Offer Price']
        all_features = ipo_features + nifty_feature_cols

        X = df[all_features].apply(pd.to_numeric, errors='coerce')
        X = X.fillna(X.mean())

        df['Listing Gain'] = pd.to_numeric(df['Listing Gain'], errors='coerce')
        df = df.dropna(subset=['Listing Gain'])
        y = (df['Listing Gain'] > 0).astype(int)
        X = X.loc[y.index]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # Class weights to handle imbalance
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        class_weight = {0: pos / (neg + pos), 1: neg / (neg + pos)}

        # Build model
        n_features = X_train_scaled.shape[1]
        model = keras.Sequential([
            layers.Input(shape=(n_features,)),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        print("  Training neural network...")
        model.fit(
            X_train_scaled, y_train,
            epochs=150,
            batch_size=16,
            validation_split=0.2,
            class_weight=class_weight,
            callbacks=[early_stop],
            verbose=0  # Silent training
        )

        y_prob = model.predict(X_test_scaled, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        acc    = round(accuracy_score(y_test, y_pred) * 100, 2)
        auc    = round(roc_auc_score(y_test, y_prob), 4)

        print(f"Model trained successfully ✅  (Accuracy: {acc}% | ROC AUC: {auc})")
        print()
        print("===== MODEL PERFORMANCE =====")
        print(f"Accuracy : {acc}%")
        print(f"ROC AUC  : {auc}")
        print(f"Layers   : 64 → 32 → 16 → 1")
        print()
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print()
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["No Gain", "Gain"]))

        return model, scaler, all_features

    except Exception as e:
        print(f"❌ Training Error: {e}")
        return None, None, None


def make_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


def parse_date(text, ref_year):
    month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                 "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    text = text.strip()
    text = ''.join(c if ord(c) < 128 else ' ' for c in text)
    text = ' '.join(text.split())
    try:
        parts = text.replace(',', '').split()
        if len(parts) == 3 and parts[0] in month_map:
            return datetime(int(parts[2]), month_map[parts[0]], int(parts[1])).date()
        if len(parts) == 3 and parts[1] in month_map:
            return datetime(int(parts[2]), month_map[parts[1]], int(parts[0])).date()
        if len(parts) == 2 and parts[0] in month_map:
            return datetime(ref_year, month_map[parts[0]], int(parts[1])).date()
        if len(parts) == 2 and parts[1] in month_map:
            return datetime(ref_year, month_map[parts[1]], int(parts[0])).date()
    except:
        pass
    return None


def to_float(text):
    text = str(text).replace(',', '')
    m = re.search(r'\d+(?:\.\d+)?', text)
    return float(m.group()) if m else None


def extract_subscription_multiple(cells):
    if len(cells) < 2:
        return None
    raw = cells[1]
    if ',' in raw:
        return None
    val = to_float(raw)
    if val is not None and 0 <= val <= 5000:
        return val
    return None


def name_keywords(ipo_name):
    skip = {'ipo', 'and', 'the', 'ltd', 'limited', 'pvt', 'private',
            'india', 'industries', 'solutions', 'services', 'technologies'}
    words = re.findall(r'[a-z0-9]+', ipo_name.lower())
    keywords = [w for w in words if w not in skip and len(w) >= 2]
    return keywords[:2]


def href_matches(ipo_name, href_slug):
    return all(kw in href_slug for kw in name_keywords(ipo_name))


def collect_all_ipo_links(driver):
    links = []
    for link in driver.find_elements(By.XPATH, "//a[@href]"):
        href = link.get_attribute("href") or ""
        m = re.search(r'/ipo/([\w-]+)/(\d+)/', href)
        if m:
            links.append((m.group(1), m.group(2)))
    return links


def subscription_warning(total, open_date, close_date, today):
    total_days  = (close_date - open_date).days + 1
    elapsed     = (today - open_date).days + 1
    pct_through = elapsed / total_days * 100
    if elapsed == 1 and total_days > 1:
        return (f"⚠️  Day 1 of {total_days} — IPO just opened today. "
                f"Subscription data not meaningful yet. "
                f"Re-run on closing day for accurate prediction.")
    if total < 1.0 and pct_through < 80:
        return (f"⚠️  WARNING: Only Day {elapsed}/{total_days} of subscription. "
                f"Total is {total}x — too early to judge. "
                f"Re-run on closing day for accurate prediction.")
    elif total < 1.0:
        return f"⚠️  Low subscription ({total}x) — market skepticism, risky."
    return None


def get_open_ipos_from_calendar(driver, today):
    month_names = ["january","february","march","april","may","june",
                   "july","august","september","october","november","december"]
    url = f"https://www.chittorgarh.com/calendar/ipo-calendar-{month_names[today.month-1]}-{today.year}/1/"
    print(f"  Loading mainboard calendar: {url}")
    driver.get(url)
    time.sleep(5)

    ipo_events = {}
    for line in driver.find_element(By.TAG_NAME, "body").text.splitlines():
        line = line.strip()
        for keyword, key in [("Opens on", "open"), ("Closes on", "close")]:
            if keyword in line:
                parts = line.split(f" {keyword} ", 1)
                if len(parts) == 2:
                    name = re.sub(r'\s+IPO$', '', parts[0].strip())
                    d = parse_date(parts[1].strip(), today.year)
                    if d:
                        ipo_events.setdefault(name, {})[key] = d

    open_ipos = []
    print()
    print("  Mainboard IPO events this month:")
    for name, dates in ipo_events.items():
        open_d, close_d = dates.get('open'), dates.get('close')
        if open_d and close_d:
            status = "✅ OPEN TODAY" if open_d <= today <= close_d else "—"
            print(f"    {name}: {open_d} → {close_d}  {status}")
            if open_d <= today <= close_d:
                open_ipos.append({"name": name, "open": open_d, "close": close_d})

    return open_ipos


def get_ipo_id(driver, ipo_name):
    keywords = name_keywords(ipo_name)
    print(f"    Searching for '{ipo_name}' (keywords: {keywords})...")

    for page_url in [
        "https://www.chittorgarh.com/ipo/ipo_dashboard.asp",
        "https://www.chittorgarh.com/report/upcoming-ipo/11/",
        "https://www.chittorgarh.com/report/ipo-subscription-status-live-bidding-data-bse-nse/21/",
    ]:
        driver.get(page_url)
        time.sleep(5)
        for slug, ipo_id in collect_all_ipo_links(driver):
            if href_matches(ipo_name, slug):
                print(f"    ✅ Found: https://www.chittorgarh.com/ipo/{slug}/{ipo_id}/")
                return slug, ipo_id

    print(f"    ❌ Could not find IPO ID for: {ipo_name} — skipping (IPO may not be active yet)")
    return None, None


def scrape_detail_page(driver, slug, ipo_id):
    url = f"https://www.chittorgarh.com/ipo/{slug}/{ipo_id}/"
    print(f"    Detail page: {url}")
    driver.get(url)
    time.sleep(4)

    issue_size = offer_price = None
    for row in driver.find_elements(By.XPATH, "//table//tr"):
        cells = [c.text.strip() for c in row.find_elements(By.TAG_NAME, "td")]
        if len(cells) < 2:
            continue
        label, value = cells[0].strip(), cells[1].strip()
        if label == 'Price Band':
            m = re.search(r'₹?([\d,]+)\s*to\s*₹?([\d,]+)', value)
            if m: offer_price = to_float(m.group(2))
        elif label == 'Total Issue Size':
            m = re.search(r'₹([\d,]+(?:\.\d+)?)\s*Cr', value)
            if m: issue_size = to_float(m.group(1))

    return issue_size, offer_price


def scrape_subscription(driver, slug, ipo_id):
    url = f"https://www.chittorgarh.com/ipo_subscription/{slug}/{ipo_id}/"
    print(f"    Subscription page: {url}")
    driver.get(url)
    time.sleep(4)

    qib = hni = rii = total = None
    for row in driver.find_elements(By.XPATH, "//table//tr"):
        cells = [c.text.strip() for c in row.find_elements(By.TAG_NAME, "td")]
        if len(cells) < 2:
            continue
        label = cells[0].strip()
        val   = extract_subscription_multiple(cells)
        if val is None:
            continue
        if label == 'QIB (Ex Anchor)':                    qib   = val
        elif label == 'NII':                               hni   = val
        elif label == 'Retail':                            rii   = val
        elif label in ('Total', 'Total **'):               total = val
        elif 'qualified institutional' in label.lower():  qib   = qib   or val
        elif 'non institutional'        in label.lower(): hni   = hni   or val
        elif 'retail individual'        in label.lower(): rii   = rii   or val
        elif 'total subscription'       in label.lower(): total = total or val

    return qib or 0, hni or 0, rii or 0, total or 0


def scrape_ipo_data(driver, ipo_name):
    slug, ipo_id = get_ipo_id(driver, ipo_name)
    if not slug or not ipo_id:
        return None

    issue_size, offer_price = scrape_detail_page(driver, slug, ipo_id)
    qib, hni, rii, total    = scrape_subscription(driver, slug, ipo_id)

    result = {
        'Issue_Size(crores)': issue_size  or 1000,
        'Offer Price':        offer_price or 500,
        'QIB': qib, 'HNI': hni, 'RII': rii, 'Total': total,
    }
    print(f"    ✅ Size: ₹{result['Issue_Size(crores)']} Cr | Price: ₹{result['Offer Price']} | "
          f"QIB: {qib}x | HNI: {hni}x | RII: {rii}x | Total: {total}x")
    return result


def run():
    nifty, nifty_feature_cols = get_nifty_features()

    model, scaler, all_features = train_ipo_model(nifty, nifty_feature_cols)
    if not model:
        return

    latest_nifty = nifty[nifty_feature_cols].iloc[-1]

    driver = make_driver()
    today  = datetime.now().date()

    print()
    print(f"📅 Today: {today}")
    print("📈 Market Context:")
    print(f"   RSI          : {round(float(latest_nifty['Nifty_RSI']), 1)}")
    print(f"   5d Return    : {round(float(latest_nifty['Nifty_Ret_5d']), 2)}%")
    print(f"   20d Return   : {round(float(latest_nifty['Nifty_Ret_20d']), 2)}%")
    print(f"   50d Return   : {round(float(latest_nifty['Nifty_Ret_50d']), 2)}%")
    print(f"   Volatility   : {round(float(latest_nifty['Nifty_Vol_20d']), 2)}%")
    print(f"   Above MA50   : {'Yes' if latest_nifty['Above_MA50'] else 'No'}")
    print(f"   Above MA200  : {'Yes' if latest_nifty['Above_MA200'] else 'No'}")
    print("─" * 60)

    print("→ Finding open mainboard IPOs from calendar...")
    open_ipos = get_open_ipos_from_calendar(driver, today)

    if not open_ipos:
        print()
        print("No open mainboard IPO detected today ❌")
        driver.quit()
        return

    print()
    print(f"→ Found {len(open_ipos)} open IPO(s). Scraping live data...")
    print("─" * 60)

    results = []
    for ipo in open_ipos:
        print(f"  🔍 {ipo['name']}")
        data = scrape_ipo_data(driver, ipo['name'])

        if not data:
            print(f"  ⏭️  Skipping {ipo['name']} — no live data available")
            continue

        live_input = pd.DataFrame([{
            'Issue_Size(crores)': data['Issue_Size(crores)'],
            'QIB':                data['QIB'],
            'HNI':                data['HNI'],
            'RII':                data['RII'],
            'Total':              data['Total'],
            'Offer Price':        data['Offer Price'],
            **latest_nifty.to_dict()
        }])

        live_input  = live_input[all_features]
        live_scaled = scaler.transform(live_input)

        prob     = float(model.predict(live_scaled, verbose=0)[0][0])
        prob_pct = round(prob * 100, 2)

        if prob > 0.75:   rec = "STRONG APPLY 🚀"
        elif prob > 0.55: rec = "APPLY WITH CAUTION ⚠️"
        else:             rec = "AVOID ❌"

        warn = subscription_warning(data['Total'], ipo['open'], ipo['close'], today)

        results.append({
            **ipo,
            "prob": prob_pct, "rec": rec, "data": data, "warn": warn
        })

    driver.quit()

    if not results:
        print()
        print("No active IPOs with live data found today ❌")
        return

    print()
    print("=" * 60)
    print("📊 FINAL ANALYSIS")
    print("=" * 60)
    for r in results:
        d = r.get("data", {})
        print()
        print(f"  📌 {r['name']}")
        print(f"     Open → Close  : {r['open']} → {r['close']}")
        if d:
            print(f"     Issue Size    : ₹{d['Issue_Size(crores)']} Cr")
            print(f"     Offer Price   : ₹{d['Offer Price']}")
            print(f"     Subscription  : QIB {d['QIB']}x | HNI {d['HNI']}x | "
                  f"RII {d['RII']}x | Total {d['Total']}x")
        print(f"     Market RSI    : {round(float(latest_nifty['Nifty_RSI']), 1)}")
        print(f"     Market Trend  : {'Bullish 📈' if latest_nifty['Above_MA200'] else 'Bearish 📉'}")
        print(f"     Success Prob  : {r['prob']}%")
        print(f"     Recommendation: {r['rec']}")
        if r.get('warn'):
            print(f"     {r['warn']}")


run()