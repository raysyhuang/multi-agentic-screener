#!/bin/bash
# (1) FMP dividend-adjusted daily (stable/historical-price-eod/dividend-adjusted) 2013-06-01..2026-11-12 for symbols_2015.txt
# (2) FMP historical market cap (stable/historical-market-capitalization) 2014-01-01..2026-11-12 for symbols_2015.txt
# (3) FMP split-adjusted daily 2003-06-01..2003-10-31 for the 2005-2015 packet's fmp_symbols.txt (gap before the packet's raw/fmp start)
cd "$(dirname "$0")/.."
U=https://financialmodelingprep.com/stable
div(){ s="$1"; f="data/fmp_divadj_2013_2026/${s}.json"; [ -s "$f" ] && exit 0
 curl -s --max-time 90 "$U/historical-price-eod/dividend-adjusted?symbol=${s}&from=2013-06-01&to=2026-11-12&apikey=$FMP_API_KEY" > "$f.tmp" && mv "$f.tmp" "$f"; }
mc(){ s="$1"; f="data/fmp_mcap_2014_2026/${s}.json"; [ -s "$f" ] && exit 0
 curl -s --max-time 90 "$U/historical-market-capitalization?symbol=${s}&from=2014-01-01&to=2026-11-12&limit=5000&apikey=$FMP_API_KEY" > "$f.tmp" && mv "$f.tmp" "$f"; }
gap(){ s="$1"; f="data/fmp_px_2003_gap/${s}.json"; [ -s "$f" ] && exit 0
 curl -s --max-time 90 "$U/historical-price-eod/full?symbol=${s}&from=2003-06-01&to=2003-10-31&apikey=$FMP_API_KEY" > "$f.tmp" && mv "$f.tmp" "$f"; }
export -f div mc gap; export U
xargs -a data/symbols_2015.txt -P 6 -I{} bash -c 'div "$@"' _ {}
xargs -a data/symbols_2015.txt -P 6 -I{} bash -c 'mc "$@"' _ {}
xargs -a /workspace/w10h2-w12h4-oos-2005-2015-20261112/build/fmp_symbols.txt -P 6 -I{} bash -c 'gap "$@"' _ {}
echo DONE
