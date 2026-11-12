# Diff Massive split/stock-dividend calendar against pack month-end price ratios (own code)
import csv,gzip,math,collections,bisect
PK='/workspace/momentum432-inputs-pack-full-20261109/'
rp=collections.defaultdict(dict);syms=collections.defaultdict(set)
for r in csv.DictReader(gzip.open(PK+'rank_prices_month_end.csv.gz','rt')):
    if r['rank_price']: rp[r['engine_ticker']][r['month_end']]=float(r['rank_price'])
    s=r['price_source']
    if s.startswith('massive_grouped_close'): syms[r['engine_ticker']].add(s.split(':',1)[1])
ME=sorted({m for d in rp.values() for m in d})
# today's / successor symbols for renamed or retired engine tickers (Massive stores some old-ticker splits under the current symbol)
SUCC={'TMK':['GL'],'DISCA':['WBD'],'DISCK':['WBD'],'BLL':['BALL'],'BKNG':['PCLN'],'IR':['TT','GDI'],'TT':['IR'],
 'FB':['META'],'ANTM':['ELV'],'WLTW':['WTW'],'RE':['EG'],'FLT':['CPAY'],'COG':['CTRA'],'CDAY':['DAY'],'PKI':['RVTY'],'ABC':['COR'],
 'NLOK':['GEN'],'SYMC':['GEN','NLOK'],'FISV':['FI'],'PEAK':['DOC'],'HCP':['DOC','PEAK'],'KORS':['CPRI'],'COH':['TPR'],'LB':['BBWI'],
 'LUK':['JEF'],'DPS':['KDP'],'MHFI':['SPGI'],'GCI':['TGNA'],'UA':['UAA'],'WAG':['WBA'],'HCN':['WELL'],'WYN':['WYND'],'ZMH':['ZBH'],
 'TSO':['ANDV'],'WLP':['ANTM','ELV'],'CBG':['CBRE'],'CSC':['DXC'],'NU':['ES'],'YHOO':['AABA'],'DLPH':['APTV'],'ARNC':['HWM'],
 'FOXA':['FOXA'],'FOX':['FOX'],'GOOGL':['GOOG'],'GOOG':['GOOGL'],'BRK.B':['BRK-B'],'BF.B':['BF-B'],'HRS':['LHX'],'CELG':[],
 'JCI':['JCI'],'BBT':['TFC'],'STI':['TFC'],'RTN':['RTX'],'UTX':['RTX'],'DWDP':['DD'],'DOW':['DD'],'MON':[],'TWX':[],'CBS':['PARA','VIAC'],'VIAC':['PARA'],'VIAB':['PARA','VIAC'],
 'LLL':['LHX'],'APC':[],'HCBK':[],'Q':['IQV'],'DISH':['SATS'],'WRK':['SW'],'SNI':[],'FTI':['FTI'],'ADS':['BFH'],'XEC':['CTRA'],'MXIM':[],'KSU':[],'XLNX':[],
 'PXD':[],'CERN':[],'ATVI':[],'TWTR':[],'CTXS':[],'NLSN':[],'ABMD':[],'DRE':[],'FBHS':['FBIN'],'MRO':[],'MMC':['MRSH'],'PARA':['PSKY'],'CTL':['LUMN'],'DWDP':['DD','DOW'],'ESV':['VAL'],'GPS':['GAP'],'HFC':['DINO'],'JEC':['J'],'WYND':['TNL'],'BHGE':['BKR'],'CHK':['EXE'],'SWN':['EXE'],'NBL':[],'CCE':['CCEP'],'COV':[],'DTV':[],'HOT':[],'TEG':[],'FDO':[],'LVLT':[],'ETFC':[],'MYL':['VTRS'],'AGN':['AGN_PLC'],'ANSS':[],'GGP':['BPYU'],'DNB':[],'EVHC':[],'CFN':[],'AVP':['NATR'],'LM':[],'HES':[],'TWC':[],'SPLS':[],'SRCL':[],'SBNY':[],'SIVB':[],'FRC':[],'MNK':['MNKKQ'],'ENDP':['ENDPQ'],'PDCO':[],'PETM':[],'QEP':[],'RAI':[],'STJ':[],'SWY':[],'VAR':[],'WCG':[],'WIN':[],'MWV':['WRK'],'JWN':[],'JNPR':[],'IPG':[],'FL':[],'FLIR':[],'FTR':[],'DO':['DO'],'DNR':[],'CXO':[],'CTLT':[],'COL':[],'AET':[],'CELG':[],'SNI':[],'SYMC':['GEN']}
sp=collections.defaultdict(list)
for r in csv.DictReader(open('massive_splits_all_2013_2026.csv')):
    sp[r['ticker']].append(r)
el=collections.defaultdict(set)
for r in csv.DictReader(gzip.open(PK+'eligibility_by_signal.csv.gz','rt')):
    if r['eligible_final']=='True': el[r['engine_ticker']].add(r['signal_date'])
SIG=sorted({s for v in el.values() for s in v})
def used(t,m):
    # month m price ratio (m_prev->m) sits inside some eligible lookback (B < m <= S, W<=12)
    i=ME.index(m)
    return sorted(s for s in el.get(t,()) if s>=m and ME.index(s)-12 < i)
out=[]
for t,px in rp.items():
    cand=set([t])|syms[t]|set(SUCC.get(t,[]))
    for c in list(cand): cand|=set(SUCC.get(c,[]))
    for c in cand:
        for s in sp.get(c,[]):
            d=s['execution_date'];hf=float(s['split_from'])/float(s['split_to'])
            if abs(math.log(hf))<1e-9: continue
            j=bisect.bisect_left(ME,d)  # first ME >= d
            if j==0 or j>=len(ME): continue
            m1=ME[j];m0=ME[j-1]
            if m0 not in px or m1 not in px: continue
            r=px[m1]/px[m0]
            unapplied=abs(math.log(r)-math.log(hf))<abs(math.log(r))
            u=used(t,m1)
            out.append(dict(engine_ticker=t,massive_symbol=c,execution_date=d,split_from=s['split_from'],split_to=s['split_to'],adjustment_type=s['adjustment_type'],
                hist_factor_massive=s['historical_adjustment_factor'],price_factor_hf=round(hf,6),pack_m0=m0,pack_p0=px[m0],pack_m1=m1,pack_p1=px[m1],pack_ratio=round(r,4),
                verdict='UNAPPLIED_IN_PACK' if unapplied else 'applied_or_absorbed',n_eligible_signals_using_month=len(u),first_signal=u[0] if u else '',last_signal=u[-1] if u else ''))
out.sort(key=lambda x:(x['verdict'],x['engine_ticker'],x['execution_date']))
w=csv.DictWriter(open('split_calendar_diff.csv','w',newline=''),fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(len(out),'split events matched to pack price series')
for x in out:
    if x['verdict']=='UNAPPLIED_IN_PACK' or abs(math.log(x['price_factor_hf']))<0.1: print(x['verdict'],x['engine_ticker'],x['massive_symbol'],x['execution_date'],x['split_from'],x['split_to'],x['adjustment_type'],x['pack_p0'],x['pack_p1'],x['pack_ratio'],x['n_eligible_signals_using_month'],x['first_signal'],x['last_signal'])
