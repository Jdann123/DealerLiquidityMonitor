"""
================================================================================
DEALER LIQUIDITY ASYMMETRY MONITOR (v2)
================================================================================

Improved version that measures DOWNSIDE LIQUIDITY STRESS in options markets.

WHAT CHANGED FROM V1:
1. Filters out junk (options < $0.10 mid, zero bids, no volume)
2. Uses DELTA BUCKETS instead of strike distance (more stable)
3. Controls for LIQUIDITY (volume, open interest) before inferring fear
4. Gamma pinning now uses OPEN INTEREST concentration, not spreads
5. Renamed metrics to be more accurate (liquidity asymmetry, not "fear")

THE CORE IDEA:
Compare matched put/call spreads at same |delta|. If puts have wider 
spreads AFTER controlling for liquidity, that's genuine downside stress.

INSTALL:
    pip install yfinance pandas numpy scipy

RUN:
    python dealer_liquidity_monitor.py

================================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# === TERMINAL COLORS ===
class C:
    R = '\033[91m'
    G = '\033[92m'
    Y = '\033[93m'
    B = '\033[94m'
    M = '\033[95m'
    C = '\033[96m'
    W = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


def col(text, c):
    return f"{c}{text}{C.END}"


# === BLACK-SCHOLES FOR DELTA CALCULATION ===

def bs_delta(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes delta"""
    if T <= 0 or sigma <= 0:
        return 0.5 if option_type == 'call' else -0.5
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def estimate_iv(mid, S, K, T, r, option_type='call'):
    """Simple IV estimation via bisection"""
    if mid <= 0 or T <= 0:
        return 0.3
    
    low, high = 0.01, 3.0
    
    for _ in range(50):
        sigma = (low + high) / 2
        price = bs_price(S, K, T, r, sigma, option_type)
        
        if abs(price - mid) < 0.001:
            return sigma
        elif price > mid:
            high = sigma
        else:
            low = sigma
    
    return sigma


def bs_price(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes price"""
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


# === DATA FETCHING ===

def fetch_and_clean(symbol):
    """Fetch option chain and apply quality filters"""
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
        
        for exp_date in expirations[:6]:  # First 6 expiries
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
        
        # === QUALITY FILTERS ===
        original_len = len(df)
        
        # 1. Must have valid bid/ask
        df = df[(df['bid'] > 0) & (df['ask'] > 0)]
        
        # 2. Mid price must be >= $0.10 (filter penny options noise)
        df['mid'] = (df['bid'] + df['ask']) / 2
        df = df[df['mid'] >= 0.10]
        
        # 3. Must have some volume OR open interest
        df = df[(df['volume'] > 0) | (df['openInterest'] > 10)]
        
        # 4. Spread must be reasonable (< 100% of mid)
        df['spread'] = df['ask'] - df['bid']
        df['spread_pct'] = df['spread'] / df['mid'] * 100
        df = df[df['spread_pct'] < 100]
        
        print(f"  Filtered: {original_len} → {len(df)} options")
        
        # === CALCULATE DELTA ===
        r = 0.05  # risk-free rate assumption
        
        deltas = []
        for _, row in df.iterrows():
            iv = row.get('impliedVolatility', 0.3)
            if pd.isna(iv) or iv <= 0:
                iv = 0.3
            
            delta = bs_delta(spot, row['strike'], row['T'], r, iv, row['type'])
            deltas.append(delta)
        
        df['delta'] = deltas
        df['abs_delta'] = df['delta'].abs()
        
        # === DELTA BUCKETS ===
        # Group into: ATM (0.4-0.6), 25d (0.2-0.4), 10d (0.05-0.2), Wings (<0.05)
        def delta_bucket(d):
            d = abs(d)
            if d >= 0.4:
                return 'ATM'
            elif d >= 0.2:
                return '25d'
            elif d >= 0.05:
                return '10d'
            else:
                return 'wing'
        
        df['delta_bucket'] = df['delta'].apply(delta_bucket)
        
        # === LIQUIDITY SCORE ===
        # Normalize volume and OI to create liquidity score
        df['volume'] = df['volume'].fillna(0)
        df['openInterest'] = df['openInterest'].fillna(0)
        
        vol_max = df['volume'].quantile(0.95) or 1
        oi_max = df['openInterest'].quantile(0.95) or 1
        
        df['liq_score'] = (
            0.6 * (df['volume'] / vol_max).clip(0, 1) +
            0.4 * (df['openInterest'] / oi_max).clip(0, 1)
        )
        
        # === LIQUIDITY-ADJUSTED SPREAD ===
        # Higher liquidity should mean tighter spreads
        # If spread is wide despite high liquidity, that's meaningful
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


# === ANALYSIS ===

def analyze_liquidity_asymmetry(result):
    """Core analysis: compare put vs call spreads at matched deltas"""
    
    df = result['data']
    spot = result['spot']
    analysis = {}
    
    # === 1. DELTA-MATCHED ASYMMETRY ===
    # Compare puts vs calls at same |delta| bucket
    
    asymmetry_by_bucket = []
    
    for bucket in ['ATM', '25d', '10d']:
        bucket_data = df[df['delta_bucket'] == bucket]
        
        calls = bucket_data[bucket_data['type'] == 'call']
        puts = bucket_data[bucket_data['type'] == 'put']
        
        if len(calls) > 0 and len(puts) > 0:
            # Use residual spreads (liquidity-adjusted)
            call_spread = calls['spread_residual'].median()
            put_spread = puts['spread_residual'].median()
            
            # Raw spreads for reference
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
    
    # === 2. OVERALL LIQUIDITY STRESS SCORE ===
    # Weighted average of asymmetries (25d most important)
    
    weights = {'ATM': 0.2, '25d': 0.5, '10d': 0.3}
    
    if len(analysis['delta_asymmetry']) > 0:
        weighted_asym = 0
        total_weight = 0
        
        for _, row in analysis['delta_asymmetry'].iterrows():
            w = weights.get(row['bucket'], 0.1)
            weighted_asym += row['asymmetry'] * w
            total_weight += w
        
        raw_score = weighted_asym / total_weight if total_weight > 0 else 0
        
        # Scale to 0-100 (calibrated so typical range maps nicely)
        analysis['stress_score'] = min(100, max(0, 50 + raw_score * 3))
    else:
        analysis['stress_score'] = 50
    
    # === 3. TERM STRUCTURE ===
    # Asymmetry by expiry
    
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
    
    # === 4. GAMMA PINNING (OI-based, not spread-based) ===
    # Find strikes with highest open interest near ATM
    
    nearest_exp = df[df['days'] == df['days'].min()]
    atm_range = nearest_exp[abs(nearest_exp['strike'] - spot) / spot < 0.10]
    
    if len(atm_range) > 0:
        oi_by_strike = atm_range.groupby('strike')['openInterest'].sum().sort_values(ascending=False)
        top_oi_strikes = oi_by_strike.head(5)
        
        # Calculate "magnet strength" - OI concentration
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
    
    # === 5. TAIL STRESS (10d put vs 25d put) ===
    # If 10-delta puts have much wider spreads than 25-delta, tail fear is elevated
    
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


# === OUTPUT ===

def print_report(result, analysis):
    """Print formatted report"""
    
    symbol = result['symbol']
    spot = result['spot']
    
    print("\n" + "=" * 70)
    print(col(f"  DEALER LIQUIDITY ASYMMETRY MONITOR: {symbol}", C.BOLD))
    print("=" * 70)
    
    print(f"\n{col('SPOT PRICE:', C.C)} ${spot:.2f}")
    
    # Stress Score
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
    
    # Delta-Matched Asymmetry
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
    
    # Term Structure
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
    
    # Gamma Pinning
    if len(analysis['pin_strikes']) > 0:
        print(f"\n{col('LIKELY PIN STRIKES (by Open Interest):', C.C)}")
        for _, row in analysis['pin_strikes'].head(3).iterrows():
            dist = row['distance']
            direction = "above" if row['strike'] > spot else "below"
            print(f"  ${row['strike']:.0f} — {row['concentration']:.1f}% of OI ({dist:.1f} {direction} spot)")
    
    # Tail Stress
    ts = analysis['tail_stress']
    print(f"\n{col('TAIL STRESS (10d vs 25d puts):', C.C)}")
    if ts.get('elevated'):
        ratio_val = ts.get('ratio', 1)
        print(f"  {col('⚠ ELEVATED: 10d puts have ' + str(round(ratio_val, 1)) + 'x wider spreads than 25d', C.R)}")
        print(f"    Dealers charging extra for deep OTM crash protection")
    else:
        print(f"  ✓ Normal: ratio = {ts.get('ratio', 1):.2f}x")
    
    # Interpretation
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
    """Compare multiple symbols"""
    
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
    
    # Print comparison
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


# === MAIN ===

def main():
    print(col("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         DEALER LIQUIDITY ASYMMETRY MONITOR (v2)                 ║
    ║         Delta-Matched • Liquidity-Adjusted • OI-Based Pinning   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """, C.C))
    
    while True:
        print("\nOptions:")
        print("  1. Analyze single symbol")
        print("  2. Compare multiple symbols")
        print("  3. Quick scan (SPY, QQQ, IWM)")
        print("  4. Exit")
        
        choice = input("\nChoice (1-4): ").strip()
        
        if choice == '1':
            symbol = input("Symbol (e.g., SPY): ").strip().upper() or 'SPY'
            
            result = fetch_and_clean(symbol)
            if result is None:
                continue
            
            analysis = analyze_liquidity_asymmetry(result)
            print_report(result, analysis)
        
        elif choice == '2':
            symbols = input("Symbols (comma-separated): ").strip().upper()
            symbols = [s.strip() for s in symbols.split(',')]
            compare_symbols(symbols)
        
        elif choice == '3':
            compare_symbols(['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA'])
        
        elif choice == '4':
            print(col("\nGoodbye!", C.C))
            break


if __name__ == "__main__":
    main()