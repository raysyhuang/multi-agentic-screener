import csv,datetime as dt,os,glob
H=os.path.dirname(os.path.abspath(__file__))
def d_of(t): return (dt.datetime(1970,1,1)+dt.timedelta(milliseconds=int(float(t)))-dt.timedelta(hours=0)).date()
def load_daily(path):
    out={}
    for r in csv.DictReader(open(path)):
        if not r.get('t'): continue
        # Massive day bars are stamped at 00:00 US/Eastern -> 04:00/05:00 UTC same date
        out[d_of(r['t']).isoformat()]=float(r['c'])
    return out
def load_month(path):
    out={}
    for r in csv.DictReader(open(path)):
        if not r.get('t'): continue
        d=d_of(r['t'])+dt.timedelta(days=1)  # month bar stamped at month start 00:00 ET (prev day 04/05 UTC) -> shift
        out[d.strftime('%Y-%m')]=float(r['c'])
    return out
import sys,json
sys.path.insert(0,os.path.join(H,'n100tickers','src'))
RENAMES=[('SUNW','JAVA','2007-08-27'),('WFMI','WFM','2011-05-06'),('ERTS','EA','2011-12-20'),('HANS','MNST','2012-01-09'),('KFT','MDLZ','2012-10-02'),('PCLN','BKNG','2018-02-27'),('QVCA','QRTEA','2018-04-09'),('CTRP','TCOM','2019-11-05'),('SYMC','NLOK','2019-11-05'),('FB','META','2022-06-09'),('LINTA','QVCA','2014-10-07')]
SPIN_RENAMES={('KFT','MDLZ'),('NWSA','FOXA'),('QVCA','QRTEA'),('SUNW','JAVA'),('HANS','MNST')}  # spin/split-off or unclear basis -> zero-return junction
SPLIT_RATIOS=[0.04,0.1,0.125,0.2,0.25,1/3,0.5,1,2,3,4,5,8,10]
def calendar():
    q=load_daily(os.path.join(H,'raw/adj/QQQQ.csv')); q2=load_daily(os.path.join(H,'raw/adj/QQQ.csv'))
    days=sorted(set(q)|set(q2)); me={}
    for d in days: me[d[:7]]=d
    return days,[me[k] for k in sorted(me)]
def membership(month_ends):
    from nasdaq_100_ticker_history import tickers_as_of
    m={}
    for d in month_ends:
        y,mo,dd=map(int,d.split('-'))
        if y<2007: m[d]=sorted(tickers_as_of(2007,1,1))
        else: m[d]=sorted(tickers_as_of(y,mo,dd))
    return m
import bisect
def build_series():
    """returns dict symbol -> (sorted dates, closes) daily, and dict symbol->{YYYY-MM: close} for month-only symbols"""
    D={}; M={}
    for f in glob.glob(os.path.join(H,'raw/adj/*.csv')):
        D[os.path.basename(f)[:-4]]=load_daily(f)
    for f in glob.glob(os.path.join(H,'raw/month/*.csv')):
        M[os.path.basename(f)[:-4]]=load_month(f)
    for f in glob.glob(os.path.join(H,'raw/override/*.csv')):  # entity overrides, e.g. MNST_pre2012 from MWW
        D[os.path.basename(f)[:-4]]=load_daily(f)
    # stitch renames: new symbol gets predecessor history scaled at the junction (junction-day return assumed 0)
    for p,s,d in RENAMES:
        P=D.get(p) or {}; S=D.get(s) or {}
        pb=[x for x in sorted(P) if x<d]; sa=[x for x in sorted(S) if x>=d]
        if not pb or not sa: continue
        k=S[sa[0]]/P[pb[-1]]
        # pure renames: keep the real junction-day return; only remove a split-basis difference (k rounded to a split ratio).
        # KFT->MDLZ and NWSA->FOXA coincide with spin-offs -> zero-return junction (k unrounded).
        if (p,s) not in SPIN_RENAMES:
            import math
            c=min(SPLIT_RATIOS,key=lambda c:abs(math.log(k/c)))
            if abs(math.log(k/c))<0.15: k=c
        new={x:P[x]*k for x in pb}; new.update({x:S[x] for x in sa})
        D[s]=new
        # predecessor key continues with the successor's prices (scaled back to the predecessor level), so a position opened
        # under the old ticker is carried through the rename instead of being cashed out early; later reuse of the old symbol is dropped
        D[p]={x:P[x] for x in pb}; D[p].update({x:S[x]/k for x in sa})
    return D,M
def me_price(D,M,sym,d,stale_days=6):
    if sym in D:
        s=D[sym]; ks=sorted(s) if not hasattr(me_price,'_c') or sym not in me_price._c else me_price._c[sym]
        me_price.__dict__.setdefault('_c',{})[sym]=ks
        i=bisect.bisect_right(ks,d)-1
        if i<0: return None
        last=ks[i]
        if (dt.date.fromisoformat(d)-dt.date.fromisoformat(last)).days>stale_days: return None
        return s[last]
    if sym in M: return M[sym].get(d[:7])
    return None
RENAMES.append(('NWSA','FOXA','2013-07-01')); RENAMES.append(('FISV','FI','2023-06-07'))  # Fiserv moved to NYSE as FI; FISV reused from 2025-11 (dropped)
# zero-return junctions for un-adjusted distributions/spins/share-class dividends (junction-day return set to 0)
JUNCTIONS=[('GOOG','2014-04-03'),('LMCA','2014-07-24'),('LMCA','2016-04-18'),('FOXA','2019-03-19'),('FOX','2019-03-19'),
           ('EXPE','2011-12-21'),('VOD','2014-02-24'),('DISCA','2014-08-07'),('DISCA','2008-10-16'),('IACI','2008-09-19')]
FACTORS=[('WDC','2025-02-24',1/1.32335),  # SanDisk spin-off ex-date 2025-02-24; FMP pre-spin ratio 1.3233-1.3235
         # spin-offs Massive left unadjusted (ex-date return ~= F-1); F from the shared packs' corporate_actions_applied.csv
         ('ADP','2007-04-02',0.8993),('ADP','2014-10-01',0.8736),('EXC','2022-02-02',0.7081),('MAR','2011-11-22',0.9504),
         ('HON','2016-10-03',0.9944),('HON','2018-10-29',0.9704),
         # BIIB/BIVV: Nasdaq ex-date 2017-02-02 (regular-way). Massive and FMP unadjusted closes agree:
         # BIIB 2017-02-02 close 264.23; BIVV first close that day 44.98; ratio 0.5 (8-K).
         # F = P_ex / (P_ex + ratio * P_child) = 264.23/(264.23+0.5*44.98) = 0.9215614536823382.
         # Same formula as the pack (e.g. EXC 2022-02-02 used the ex-date parent close 42.86).
         # The pack listed ex_date 2017-02-01 and used the cum close 281.69, giving F=0.926063515023999.
         # FMP historical-price-eod and its dividend-adjusted series do not adjust this event.
         ('BIIB','2017-02-02',264.23/(264.23+0.5*44.98))]
GAP_OK={'JAVA','ATVI','DISCA','IACI','QRTEA'}   # same-entity data gaps of ~1 month (vendor gaps), allowed
ALIAS_BEFORE={'MNST':('2012-01-09','MNST_MonsterWorldwide')}  # Massive MNST history pre-2012 is not Monster Worldwide -> no clean data
def fixed_series():
    D,M=build_series()
    for t,d in JUNCTIONS:
        s=D[t]; ks=sorted(s); pre=[k for k in ks if k<d]; post=[k for k in ks if k>=d]
        k=s[post[0]]/s[pre[-1]]
        for x in pre: s[x]*=k
    # LMCK monthly series: apply LMCA's 2016-04-18 distribution factor (same event) to months before 2016-04
    s=D['LMCA']  # already junction-fixed; recover factor from raw file
    r=load_daily(os.path.join(H,'raw/adj/LMCA.csv')); f=r['2016-04-18']/r['2016-04-15']
    D['__LMCK_factor']={'f':f}
    M['LMCK']={k:(v*f if k<'2016-04' else v) for k,v in M['LMCK'].items()}
    del D['__LMCK_factor']
    # spin-offs Massive did not adjust (found via FMP cross-check): scale pre-ex-date closes by FMP-implied factor
    for t,d,fac in FACTORS:
        for x in D[t]:
            if x<d: D[t][x]*=fac
    # MNST pre-2012: drop (vendor history is Monster Beverage back-filled); stitched HANS already provides MNST lookback
    return D,M
def series_key(sym,d):
    if sym in ALIAS_BEFORE and d<ALIAS_BEFORE[sym][0]: return ALIAS_BEFORE[sym][1]
    return sym
class Px:
    def __init__(s):
        s.D,s.M=fixed_series(); s.k={t:sorted(v) for t,v in s.D.items()}
        # segment ids: break at gaps >10 calendar days (except GAP_OK)
        s.seg={}
        for t,ks in s.k.items():
            sid=0; seg={}
            for a,b in zip([None]+ks,ks):
                if a and (dt.date.fromisoformat(b)-dt.date.fromisoformat(a)).days>(40 if t in GAP_OK else 10): sid+=1
                seg[b]=sid
            s.seg[t]=seg
    def at(s,t,d,stale=6):
        """(price,date,segment) of last close on/before d within `stale` days"""
        if t in s.k:
            ks=s.k[t]; i=bisect.bisect_right(ks,d)-1
            if i<0: return None
            last=ks[i]; lim=35 if t in GAP_OK else stale
            if (dt.date.fromisoformat(d)-dt.date.fromisoformat(last)).days>lim: return None
            return s.D[t][last],last,s.seg[t][last]
        if t in s.M:
            v=s.M[t].get(d[:7]); return (v,d,0) if v else None
        return None
    def hold_ret(s,t,d0,d1):
        """return from close at d0 to close at d1; if delisted/gapped in between, use last close of the contiguous segment (cash-out)"""
        a=s.at(t,d0)
        if not a: return None
        if t in s.k:
            ks=s.k[t]; i=bisect.bisect_right(ks,d1)-1; last=ks[i]
            # walk back to stay within segment of d0
            while s.seg[t][last]!=a[2]: i-=1; last=ks[i]
            return s.D[t][last]/a[0]-1, last
        b=s.M[t].get(d1[:7])
        if b is None:  # delisted mid-month: last monthly bar
            ms=[k for k in sorted(s.M[t]) if k<=d1[:7] and k>=d0[:7]]; b=s.M[t][ms[-1]]
        return b/a[0]-1, d1
    def daily_path(s,t,d0,d1,days):
        """daily closes (relative to d0) for days in (d0,d1], forward-filled, within segment"""
        a=s.at(t,d0); out=[]
        if t not in s.k:
            r=s.hold_ret(t,d0,d1)[0]; return [1.0]*(len(days)-1)+[1+r]
        last=1.0; seg=a[2]
        for d in days:
            v=s.D[t].get(d)
            if v is not None and s.seg[t][d]==seg: last=v/a[0]
            out.append(last)
        return out
