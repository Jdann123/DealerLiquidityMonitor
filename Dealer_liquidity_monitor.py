"""
DEALER LIQUIDITY ASYMMETRY MONITOR
===================================

What this does:
    Market makers widen bid-ask spreads when they're nervous.
    They widen puts MORE than calls if they're scared of a drop.
    This tool detects that asymmetry using real options data.

The trick:
    You can't just compare raw spreads — illiquid options always look "scared."
    So we compare puts vs calls at the SAME delta, and adjust for volume/OI.
    What's left over is the real signal.

To run:
    pip install yfinance pandas numpy scipy
    python dealer_liquidity_monitor.py

"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# --- Terminal colors (makes output pretty) ---
class C:
    R = '\033[91m'      # red
    G = '\033[92m'      # green
    Y = '\033[93m'      # yellow
    B = '\033[94m'      # blue
    M = '\033[95m'      # magenta
    C = '\033[96m'      # cyan
    W = '\033[97m'      # white
    BOLD = '\033[1m'
    END = '\033[0m'


def col(text, c):
    """Wrap text in color codes"""
    return f"{c}{text}{C.END}"


# --- Black-Scholes math (need this to calculate delta) ---

def bs_delta(S, K, T, r, sigma, option_type='call'):
    """
    How much does the option move when the stock moves $1?
    Delta ranges from 0 to 1 for calls, 0 to -1 for puts.
    ATM options have delta around 0.5 (or -0.5 for puts).
    """
    if T <= 0 or sigma <= 0:
        return 0.5 if option_type == 'call' else -0.5
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def bs_price(S, K, T, r, sigma, option_type='call'):
    """Standard Black-Scholes pricing formula"""
    if T <= 0:
        if option_type == 'call':
            return max(0, S - K)
        else:
            return max(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# --- Fetching and cleaning data ---

def fetch_and_clean(symbol):
    """
    Grab the option chain from Yahoo Finance and clean it up.
    We filter out garbage: penny options, zero bids, no volume, etc.
    """
    print(f"\n{col('Fetching ' + symbol + '...', C.C)}")
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        spot = info.get('regularMarketPrice') or info.get('previousClose', 100)
        
        print(f"  Spot: ${spot:.2f}")
        
        expirations = ticker.options
        if not expirations:
            print(col("  No options available", C.R))
            return None
        
        all_data = []
        
        # Grab first 6 expiries (don't need more for this)
        for exp_date in expirations[:6]:
            try:
                chain = ticker.option_chain(exp_date)
                
                for opt_type, df in [('call', chain.calls), ('put', chain.puts)]:
                    df = df.copy()
                    df['type'] = opt_type
                    df['expiry'] = exp_date
                    
                    exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                    df['days'] = max(1, (exp_dt - datetime.now()).days)
                    df['T'] = df['days'] / 365
                    
                    all_data.append(df)
                
                days = (datetime.strptime(exp_date, '%Y-%m-%d') - datetime.now()).days
                print(f"  ✓ {exp_date} ({days}d)")
                
            except Exception as e:
                continue
        
        if not all_data:
            return None
        
        df = pd.concat(all_data, ignore_index=True)
        
        # --- Filter out the junk ---
        original_len = len(df)
        
        # Need real bid/ask
        df = df[(df['bid'] > 0) & (df['ask'] > 0)]
        
        # No penny options (super noisy)
        df['mid'] = (df['bid'] + df['ask']) / 2
        df = df[df['mid'] >= 0.10]
        
        # Need SOME activity
        df = df[(df['volume'] > 0) | (df['openInterest'] > 10)]
        
        # Spread can't be insane
        df['spread'] = df['ask'] - df['bid']
        df['spread_pct'] = df['spread'] / df['mid'] * 100
        df = df[df['spread_pct'] < 100]
        
        print(f"  Filtered: {original_len} → {len(df)} options")
        
        # --- Calculate delta for each option ---
        r = 0.05  # just assume 5% rate, close enough
        
        deltas = []
        for _, row in df.iterrows():
            iv = row.get('impliedVolatility', 0.3)
            if pd.isna(iv) or iv <= 0:
                iv = 0.3  # fallback
            
            delta = bs_delta(spot, row['strike'], row['T'], r, iv, row['type'])
            deltas.append(delta)
        
        df['delta'] = deltas
        df['abs_delta'] = df['delta'].abs()
        
        # --- Group into delta buckets ---
        # This is key: we compare 25-delta puts to 25-delta calls, not by strike
        def delta_bucket(d):
            d = abs(d)
            if d >= 0.4:
                return 'ATM'      # at-the-money
            elif d >= 0.2:
                return '25d'      # 25 delta (typical hedge level)
            elif d >= 0.05:
                return '10d'      # 10 delta (tail options)
            else:
                return 'wing'     # super far OTM, ignore these
        
        df['delta_bucket'] = df['delta'].apply(delta_bucket)
        
        # --- Liquidity score ---
        # More volume + OI = more liquid = should have tighter spreads
        df['volume'] = df['volume'].fillna(0)
        df['openInterest'] = df['openInterest'].fillna(0)
        
        vol_max = df['volume'].quantile(0.95) or 1
        oi_max = df['openInterest'].quantile(0.95) or 1
        
        df['liq_score'] = (
            0.6 * (df['volume'] / vol_max).clip(0, 1) +
            0.4 * (df['openInterest'] / oi_max).clip(0, 1)
        )
        
        # --- Expected spread (what spread "should" be given liquidity) ---
        # If actual spread is wider than expected, that's the signal
        df['expected_spread'] = 5 + 20 * (1 - df['liq_score']) + 10 * (1 - df['abs_delta'])
        df['spread_residual'] = df['spread_pct'] - df['expected_spread']
        
        return {
            'symbol': symbol,
            'spot': spot,
            'data': df,
            'expirations': expirations[:6]
        }
        
    except Exception as e:
        print(col(f"  Error: {e}", C.R))
        return None


# --- The actual analysis ---

def analyze_liquidity_asymmetry(result):
    """
    This is where the magic happens.
    We compare put spreads vs call spreads at each delta bucket.
    If puts are wider (after adjusting for liquidity), dealers are scared of downside.
    """
    
    df = result['data']
    spot = result['spot']
    analysis = {}
    
    # --- Compare puts vs calls at each delta level ---
    asymmetry_by_bucket = []
    
    for bucket in ['ATM', '25d', '10d']:
        bucket_data = df[df['delta_bucket'] == bucket]
        
        calls = bucket_data[bucket_data['type'] == 'call']
        puts = bucket_data[bucket_data['type'] == 'put']
        
        if len(calls) > 0 and len(puts) > 0:
            # Use liquidity-adjusted spreads
            call_spread = calls['spread_residual'].median()
            put_spread = puts['spread_residual'].median()
            
            # Also track raw spreads for reference
            call_raw = calls['spread_pct'].median()
            put_raw = puts['spread_pct'].median()
            
            asymmetry_by_bucket.append({
                'bucket': bucket,
                'call_spread_raw': call_raw,
                'put_spread_raw': put_raw,
                'call_spread_adj': call_spread,
                'put_spread_adj': put_spread,
                'asymmetry': put_spread - call_spread,  # positive = puts wider
                'ratio': put_raw / call_raw if call_raw > 0 else 1
            })
    
    analysis['delta_asymmetry'] = pd.DataFrame(asymmetry_by_bucket)
    
    # --- Overall stress score ---
    # Weight 25d the most (that's where the action is)
    weights = {'ATM': 0.2, '25d': 0.5, '10d': 0.3}
    
    if len(analysis['delta_asymmetry']) > 0:
        weighted_asym = 0
        total_weight = 0
        
        for _, row in analysis['delta_asymmetry'].iterrows():
            w = weights.get(row['bucket'], 0.1)
            weighted_asym += row['asymmetry'] * w
            total_weight += w
        
        raw_score = weighted_asym / total_weight if total_weight > 0 else 0
        
        # Scale to 0-100
        analysis['stress_score'] = min(100, max(0, 50 + raw_score * 3))
    else:
        analysis['stress_score'] = 50
    
    # --- Term structure (how does asymmetry change by expiry?) ---
    term_structure = []
    
    for exp in df['expiry'].unique():
        exp_data = df[df['expiry'] == exp]
        calls = exp_data[exp_data['type'] == 'call']
        puts = exp_data[exp_data['type'] == 'put']
        
        if len(calls) > 0 and len(puts) > 0:
            days = exp_data['days'].iloc[0]
            
            call_spread = calls['spread_pct'].median()
            put_spread = puts['spread_pct'].median()
            
            total_oi = exp_data['openInterest'].sum()
            total_vol = exp_data['volume'].sum()
            
            term_structure.append({
                'expiry': exp,
                'days': days,
                'call_spread': call_spread,
                'put_spread': put_spread,
                'asymmetry': put_spread - call_spread,
                'ratio': put_spread / call_spread if call_spread > 0 else 1,
                'total_oi': total_oi,
                'total_volume': total_vol
            })
    
    analysis['term_structure'] = pd.DataFrame(term_structure).sort_values('days')
    
    # --- Gamma pinning (where will price "stick" on expiry?) ---
    # Look at open interest concentration, not spreads
    nearest_exp = df[df['days'] == df['days'].min()]
    atm_range = nearest_exp[abs(nearest_exp['strike'] - spot) / spot < 0.10]
    
    if len(atm_range) > 0:
        oi_by_strike = atm_range.groupby('strike')['openInterest'].sum().sort_values(ascending=False)
        top_oi_strikes = oi_by_strike.head(5)
        
        total_oi = oi_by_strike.sum()
        
        pin_strikes = []
        for strike, oi in top_oi_strikes.items():
            concentration = oi / total_oi if total_oi > 0 else 0
            pin_strikes.append({
                'strike': strike,
                'oi': oi,
                'concentration': concentration * 100,
                'distance': abs(strike - spot)
            })
        
        analysis['pin_strikes'] = pd.DataFrame(pin_strikes)
    else:
        analysis['pin_strikes'] = pd.DataFrame()
    
    # --- Tail stress (are deep OTM puts extra wide?) ---
    # Compare 10-delta puts to 25-delta puts
    # If 10d are much wider, dealers are really scared of a crash
    puts_25d = df[(df['type'] == 'put') & (df['delta_bucket'] == '25d')]
    puts_10d = df[(df['type'] == 'put') & (df['delta_bucket'] == '10d')]
    
    if len(puts_25d) > 0 and len(puts_10d) > 0:
        spread_25d = puts_25d['spread_pct'].median()
        spread_10d = puts_10d['spread_pct'].median()
        
        analysis['tail_stress'] = {
            'spread_25d': spread_25d,
            'spread_10d': spread_10d,
            'ratio': spread_10d / spread_25d if spread_25d > 0 else 1,
            'elevated': spread_10d / spread_25d > 1.5 if spread_25d > 0 else False
        }
    else:
        analysis['tail_stress'] = {'ratio': 1, 'elevated': False}
    
    return analysis


# --- Print the results nicely ---

def print_report(result, analysis):
    """Format and print everything"""
    
    symbol = result['symbol']
    spot = result['spot']
    
    print("\n" + "=" * 70)
    print(col(f"  DEALER LIQUIDITY ASYMMETRY MONITOR: {symbol}", C.BOLD))
    print("=" * 70)
    
    print(f"\n{col('SPOT PRICE:', C.C)} ${spot:.2f}")
    
    # Stress score with color coding
    score = analysis['stress_score']
    if score > 65:
        score_col = C.R
        label = "ELEVATED DOWNSIDE STRESS"
    elif score > 50:
        score_col = C.Y
        label = "MODERATE ASYMMETRY"
    else:
        score_col = C.G
        label = "BALANCED LIQUIDITY"
    
    print(f"\n{col('LIQUIDITY STRESS SCORE:', C.C)} {col(f'{score:.0f}/100', score_col)} ({label})")
    
    # Visual gauge
    gauge_len = 40
    filled = int(score / 100 * gauge_len)
    gauge = "█" * filled + "░" * (gauge_len - filled)
    print(f"  [{gauge}]")
    
    # Delta-matched asymmetry table
    print(f"\n{col('DELTA-MATCHED SPREAD ASYMMETRY:', C.C)}")
    print(f"  {'Bucket':<8} {'Call Spd%':<12} {'Put Spd%':<12} {'Asymmetry':<12} {'Interpretation'}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*20}")
    
    for _, row in analysis['delta_asymmetry'].iterrows():
        asym = row['asymmetry']
        if asym > 3:
            interp = col("puts wider ↓", C.R)
        elif asym < -3:
            interp = col("calls wider ↑", C.Y)
        else:
            interp = col("balanced", C.G)
        
        print(f"  {row['bucket']:<8} {row['call_spread_raw']:<12.1f} {row['put_spread_raw']:<12.1f} {asym:<+12.1f} {interp}")
    
    # Term structure
    print(f"\n{col('TERM STRUCTURE:', C.C)}")
    print(f"  {'Expiry':<12} {'Days':<6} {'P/C Ratio':<10} {'OI':<12} {'Status'}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*12} {'-'*15}")
    
    for _, row in analysis['term_structure'].iterrows():
        ratio = row['ratio']
        if ratio > 1.3:
            status = col("stressed", C.R)
        elif ratio > 1.1:
            status = col("tilted", C.Y)
        else:
            status = col("normal", C.G)
        
        print(f"  {row['expiry']:<12} {row['days']:<6.0f} {ratio:<10.2f} {row['total_oi']:<12,.0f} {status}")
    
    # Pin strikes
    if len(analysis['pin_strikes']) > 0:
        print(f"\n{col('LIKELY PIN STRIKES (by Open Interest):', C.C)}")
        for _, row in analysis['pin_strikes'].head(3).iterrows():
            dist = row['distance']
            direction = "above" if row['strike'] > spot else "below"
            print(f"  ${row['strike']:.0f} — {row['concentration']:.1f}% of OI ({dist:.1f} {direction} spot)")
    
    # Tail stress
    ts = analysis['tail_stress']
    print(f"\n{col('TAIL STRESS (10d vs 25d puts):', C.C)}")
    if ts.get('elevated'):
        ratio_val = ts.get('ratio', 1)
        print(f"  {col('⚠ ELEVATED: 10d puts have ' + str(round(ratio_val, 1)) + 'x wider spreads than 25d', C.R)}")
        print(f"    Dealers charging extra for deep OTM crash protection")
    else:
        print(f"  ✓ Normal: ratio = {ts.get('ratio', 1):.2f}x")
    
    # Bottom line interpretation
    print(f"\n{col('INTERPRETATION:', C.C)}")
    print("─" * 50)
    
    if score > 65:
        print(col("  ⚠ Downside liquidity is stressed.", C.R))
        print("    Put spreads are significantly wider than calls at matched deltas.")
        print("    This persists AFTER controlling for liquidity differences.")
        print("    Implication: dealers are charging more for downside exposure.")
    elif score > 50:
        print(col("  ◐ Slight downside tilt in liquidity.", C.Y))
        print("    Some asymmetry present but within normal ranges.")
    else:
        print(col("  ✓ Liquidity is balanced across puts and calls.", C.G))
        print("    No significant asymmetry after controlling for delta and liquidity.")
    
    print("\n" + "=" * 70)


def compare_symbols(symbols):
    """Run analysis on multiple symbols and show side-by-side"""
    
    results = []
    
    for sym in symbols:
        result = fetch_and_clean(sym)
        if result is None:
            continue
        
        analysis = analyze_liquidity_asymmetry(result)
        
        results.append({
            'symbol': sym,
            'spot': result['spot'],
            'stress_score': analysis['stress_score'],
            'tail_stress': analysis['tail_stress'].get('ratio', 1),
            'avg_asymmetry': analysis['delta_asymmetry']['asymmetry'].mean() if len(analysis['delta_asymmetry']) > 0 else 0
        })
    
    # Print comparison table
    print("\n" + "=" * 70)
    print(col("  MULTI-SYMBOL COMPARISON", C.BOLD))
    print("=" * 70)
    
    print(f"\n  {'Symbol':<8} {'Spot':<10} {'Stress':<10} {'Tail':<10} {'Asymmetry'}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for r in sorted(results, key=lambda x: x['stress_score'], reverse=True):
        score_col = C.R if r['stress_score'] > 65 else C.Y if r['stress_score'] > 50 else C.G
        stress_val = r['stress_score']
        print(f"  {r['symbol']:<8} ${r['spot']:<9.2f} {col(str(int(stress_val)), score_col):<18} {r['tail_stress']:<10.2f} {r['avg_asymmetry']:<+10.1f}")
    
    return results


# --- Main menu ---

def main():
    print(col("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         DEALER LIQUIDITY ASYMMETRY MONITOR                      ║
    ║         Detects downside stress from options spread patterns     ║
    ╚══════════════════════════════════════════════════════════════════╝
    """, C.C))
    
    while True:
        print("\nWhat do you want to do?")
        print("  1. Analyze a single stock")
        print("  2. Compare multiple stocks")
        print("  3. Quick scan (SPY, QQQ, IWM, AAPL, TSLA)")
        print("  4. Exit")
        
        choice = input("\nChoice (1-4): ").strip()
        
        if choice == '1':
            symbol = input("Enter symbol (e.g., SPY): ").strip().upper() or 'SPY'
            
            result = fetch_and_clean(symbol)
            if result is None:
                continue
            
            analysis = analyze_liquidity_asymmetry(result)
            print_report(result, analysis)
        
        elif choice == '2':
            symbols = input("Enter symbols separated by comma: ").strip().upper()
            symbols = [s.strip() for s in symbols.split(',')]
            compare_symbols(symbols)
        
        elif choice == '3':
            compare_symbols(['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA'])
        
        elif choice == '4':
            print(col("\nLater!", C.C))
            break


if __name__ == "__main__":
    main()