"""Parse Massive /v2/aggs/ticker/{T}/range/1/day (adjusted=true) dumps (CSV v,vw,o,c,h,l,t,n) into
data/massive_daily/<era>_<ENGINE>.csv (date,close,massive_ticker). Dates = t in America/New_York."""
import csv, re, os, datetime as dt, sys
from zoneinfo import ZoneInfo
NY = ZoneInfo('America/New_York')
A = os.environ.get('MASSIVE_DUMPS', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'massive_daily', 'raw_dumps')) + '/'
M = [('2005','FDO','FDO','3ee40d9c'),('2005','ESRX','ESRX','b49ea863'),('2005','ARNC','AA','fdd8c021'),('2015','ARNC','AA','fdd8c021'),
     ('2005','ASH','ASH','ea145650'),('2005','BIG','BIG','9e389c49'),('2005','BLS','BLS','5d3bb393'),('2005','ETFC','ETFC','7740f4e0'),
     ('2005','FITB','FITB','ebbc9f40'),('2005','IVZ','IVZ','687ee4c6'),('2005','LEHMQ','LEH','29122aa0'),('2005','LSI','LSI','13545ce8'),
     ('2005','MHS','MHS','29b71f81'),('2005','MON','MON','9050dd82'),('2005','Q','Q','a8f4089d'),('2005','SCG','SCG','e5850616'),
     ('2005','SGP','SGP','bbef2cac'),('2005','TIE','TIE','b1b90d7c'),('2005','TXU','TXU','a12e9add'),('2005','WYE','WYE','a53385a1'),
     ('2005','WFM','WFM','e54b79f9'),('2015','CHTR','CHTR','fef07e09'),('2015','DXC','CSC','e151cfbe'),('2015','DXC','DXC','23b3b669'),
     ('2015','GE','GE','05806e19'),('2015','INFO','INFO','923f14bd'),('2015','NI','NI','a9202204'),('2015','WYND','WYN','64360f4d'),
     ('2005','WFM','WFMI','9babcd21'),('2005','ETFC','ET','ca125fd7'),('2005','BIG','BLI','f9cac2ee'),
     ('2005','BMC','BMC','7cc3d6cf'),('2005','SNDK','SNDK','12a478d3'),
     ('2005','KMG','KMG','b4e20294'),('2005','EP','EP','761f0161'),('2005','MIL','MIL','1b05d68a'),('2005','TLAB','TLAB','c60e7bd9'),
     ('2005','GR','GR','6d03c8f2'),('2005','CVH','CVH','0afcb1d6'),
     ('2005','BF.B','BF.B','13bd07db'),('2005','BKNG','PCLN','a2fca8c5'),('2005','EMC','EMC','2638fea8'),('2005','FOXA','NWSA','b3dc259f'),('2005','FOXA','NWS.A','911cc628'),
     ('2005','LLL','LLL','6565d605'),('2005','NBL','NBL','96522541'),('2005','S','S','64294700'),('2005','SUN','SUN','e77288cc'),
     ('2005','TE','TE','4b5d98f9'),('2005','TMUS','PCS','a5e5b9ed'),('2005','XL','XL','63699e92'),('2005','UST','UST','a6ab572e')]
out = {}
for era, eng, mt, h in M:
    fs = [f for f in os.listdir(A) if f.startswith(h)]
    assert len(fs) == 1, (h, fs)
    src = open(A + fs[0]).read().split('Structured content')[0]
    for l in src.split('\n'):
        p = l.strip().split(',')
        if len(p) == 8 and re.match(r'^[0-9.]+$', p[0]) and p[6].isdigit():
            d = dt.datetime.fromtimestamp(int(p[6]) / 1000, NY).date().isoformat()
            out.setdefault((era, eng), {})[d] = (float(p[3]), mt)
# JAVA (Sun Microsystems): stored via call_api store_as + query_data group_concat (t:c pairs); first occurrence kept
src = open(A + 'JAVA_query_data_groupconcat.txt').read()
seen = {}
for t, c in re.findall(r'(\d{13}):([0-9.]+);', src):
    d = dt.datetime.fromtimestamp(int(t) / 1000, NY).date().isoformat()
    if d not in seen: seen[d] = (float(c), 'JAVA')
out[('2005', 'JAVA')] = seen
os.makedirs('data/massive_daily', exist_ok=True)
for (era, eng), rows in out.items():
    with open('data/massive_daily/%s_%s.csv' % (era, eng), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['date', 'close', 'massive_ticker'])
        for d in sorted(rows): w.writerow([d, rows[d][0], rows[d][1]])
    ds = sorted(rows); print(era, eng, len(ds), ds[0], ds[-1])
