# Same-Day Duplicate & SSIM Leakage Audit Report

## Decision: **Significant Leakage: 64.3% near-duplicates (> 30%). STOP and report to human before TTA.**

### Executive Summary
- **Total Validation Images**: `115`
- **0-Day Gap Pairs Analyzed**: `93`
- **Mean SSIM Across All Pairs**: `0.9136`

### Category Breakdown
| Category | SSIM Range | Count | % of Total Val Set | Leakage Severity |
|:---|:---:|:---:|:---:|:---|
| **Near-Duplicate** | `> 0.95` | **74** | **64.3%** | Real Leakage |
| **Similar but Different** | `0.80 - 0.95` | **0** | **0.0%** | Marginal (Different observation hours) |
| **Clearly Different** | `< 0.80` | **19** | **16.5%** | Not Leakage |

### List of Near-Duplicate Validation Images (SSIM > 0.95)
| # | Validation Image | Nearest Training Image | SSIM | Time Gap (Hours) |
|:---:|:---|:---|:---:|:---:|
| 1 | `20170425133730Ch.jpeg` | `20170425133730Ch.jpeg` | **1.0000** | 0.00h |
| 2 | `20130425135954Bh.jpeg` | `20130425135954Bh.jpeg` | **1.0000** | 0.00h |
| 3 | `20160315133714Th.jpeg` | `20160315133714Th.jpeg` | **1.0000** | 0.00h |
| 4 | `20160213153054Bh.jpeg` | `20160213153054Bh.jpeg` | **1.0000** | 0.00h |
| 5 | `20141210201814Mh.jpeg` | `20141210201814Mh.jpeg` | **1.0000** | 0.00h |
| 6 | `20170119031254Uh.jpeg` | `20170119031254Uh.jpeg` | **1.0000** | 0.00h |
| 7 | `20110410141654Bh.jpeg` | `20110410141654Bh.jpeg` | **1.0000** | 0.00h |
| 8 | `20170117175014Mh.jpeg` | `20170117175014Mh.jpeg` | **1.0000** | 0.00h |
| 9 | `20161231133734Ch.jpeg` | `20161231133734Ch.jpeg` | **1.0000** | 0.00h |
| 10 | `20110327143554Bh.jpeg` | `20110327143554Bh.jpeg` | **1.0000** | 0.00h |
| 11 | `20170418013450Uh.jpeg` | `20170418013450Uh.jpeg` | **1.0000** | 0.00h |
| 12 | `20111027145954Bh.jpeg` | `20111027145954Bh.jpeg` | **1.0000** | 0.00h |
| 13 | `20160401170314Mh.jpeg` | `20160401170314Mh.jpeg` | **1.0000** | 0.00h |
| 14 | `20181101230010Mh.jpeg` | `20181101230010Mh.jpeg` | **1.0000** | 0.00h |
| 15 | `20111009110234Ch.jpeg` | `20111009110234Ch.jpeg` | **1.0000** | 0.00h |
| 16 | `20150731082034Lh.jpeg` | `20150731082034Lh.jpeg` | **1.0000** | 0.00h |
| 17 | `20131025185414Mh.jpeg` | `20131025185414Mh.jpeg` | **1.0000** | 0.00h |
| 18 | `20140604195854Bh.jpeg` | `20140604195854Bh.jpeg` | **1.0000** | 0.00h |
| 19 | `20211225025850Uh.jpeg` | `20211225025850Uh.jpeg` | **1.0000** | 0.00h |
| 20 | `20110727063234Lh.jpeg` | `20110727063234Lh.jpeg` | **1.0000** | 0.00h |
| 21 | `20160501143154Bh.jpeg` | `20160501143154Bh.jpeg` | **1.0000** | 0.00h |
| 22 | `20150116082034Lh.jpeg` | `20150116082034Lh.jpeg` | **1.0000** | 0.00h |
| 23 | `20120814163114Mh.jpeg` | `20120814163114Mh.jpeg` | **1.0000** | 0.00h |
| 24 | `20190626025430Lh.jpeg` | `20190626025430Lh.jpeg` | **1.0000** | 0.00h |
| 25 | `20120913165314Mh.jpeg` | `20120913165314Mh.jpeg` | **1.0000** | 0.00h |
| 26 | `20170804140650Bh.jpeg` | `20170804140650Bh.jpeg` | **1.0000** | 0.00h |
| 27 | `20110128174814Mh.jpeg` | `20110128174814Mh.jpeg` | **1.0000** | 0.00h |
| 28 | `20150125172714Mh.jpeg` | `20150125172714Mh.jpeg` | **1.0000** | 0.00h |
| 29 | `20171008144250Bh.jpeg` | `20171008144250Bh.jpeg` | **1.0000** | 0.00h |
| 30 | `20170927143450Bh.jpeg` | `20170927143450Bh.jpeg` | **1.0000** | 0.00h |
| 31 | `20131006075314Th.jpeg` | `20131006075314Th.jpeg` | **1.0000** | 0.00h |
| 32 | `20140312081114Th.jpeg` | `20140312081114Th.jpeg` | **1.0000** | 0.00h |
| 33 | `20130903141654Bh.jpeg` | `20130903141654Bh.jpeg` | **1.0000** | 0.00h |
| 34 | `20150113105034Ch.jpeg` | `20150113105034Ch.jpeg` | **1.0000** | 0.00h |
| 35 | `20110823120334Ch.jpeg` | `20110823120334Ch.jpeg` | **1.0000** | 0.00h |
| 36 | `20171115222230Lh.jpeg` | `20171115222230Lh.jpeg` | **1.0000** | 0.00h |
| 37 | `20120113104834Ch.jpeg` | `20120113104834Ch.jpeg` | **1.0000** | 0.00h |
| 38 | `20120723095014Th.jpeg` | `20120723095014Th.jpeg` | **1.0000** | 0.00h |
| 39 | `20141016022554Uh.jpeg` | `20141016022554Uh.jpeg` | **1.0000** | 0.00h |
| 40 | `20121027165114Mh.jpeg` | `20121027165114Mh.jpeg` | **1.0000** | 0.00h |
| 41 | `20121129063154Uh.jpeg` | `20121129063154Uh.jpeg` | **1.0000** | 0.00h |
| 42 | `20121129063154Uh.jpeg` | `20121129063154Uh.jpeg` | **1.0000** | 0.00h |
| 43 | `20121027063134Lh.jpeg` | `20121027063134Lh.jpeg` | **1.0000** | 0.00h |
| 44 | `20171011170310Mh.jpeg` | `20171011170310Mh.jpeg` | **1.0000** | 0.00h |
| 45 | `20130925143254Bh.jpeg` | `20130925143254Bh.jpeg` | **1.0000** | 0.00h |
| 46 | `20211231223430Lh.jpeg` | `20211231223430Lh.jpeg` | **1.0000** | 0.00h |
| 47 | `20120603123034Ch.jpeg` | `20120603123034Ch.jpeg` | **1.0000** | 0.00h |
| 48 | `20130811140254Bh.jpeg` | `20130811140254Bh.jpeg` | **1.0000** | 0.00h |
| 49 | `20130407063154Uh.jpeg` | `20130407063154Uh.jpeg` | **1.0000** | 0.00h |
| 50 | `20120205172414Mh.jpeg` | `20120205172414Mh.jpeg` | **1.0000** | 0.00h |
| 51 | `20131017185334Ch.jpeg` | `20131017185334Ch.jpeg` | **1.0000** | 0.00h |
| 52 | `20110322005814Mh.jpeg` | `20110322005814Mh.jpeg` | **1.0000** | 0.00h |
| 53 | `20130107174854Bh.jpeg` | `20130107174854Bh.jpeg` | **1.0000** | 0.00h |
| 54 | `20130107174854Bh.jpeg` | `20130107174854Bh.jpeg` | **1.0000** | 0.00h |
| 55 | `20110815140454Bh.jpeg` | `20110815140454Bh.jpeg` | **1.0000** | 0.00h |
| 56 | `20190926165810Mh.jpeg` | `20190926165810Mh.jpeg` | **1.0000** | 0.00h |
| 57 | `20141216222634Lh.jpeg` | `20141216222634Lh.jpeg` | **1.0000** | 0.00h |
| 58 | `20150806082034Lh.jpeg` | `20150806082034Lh.jpeg` | **1.0000** | 0.00h |
| 59 | `20210817165030Mh.jpeg` | `20210817165030Mh.jpeg` | **1.0000** | 0.00h |
| 60 | `20170103030554Uh.jpeg` | `20170103030554Uh.jpeg` | **1.0000** | 0.00h |
| 61 | `20171022170710Mh.jpeg` | `20171022170710Mh.jpeg` | **1.0000** | 0.00h |
| 62 | `20131027025654Uh.jpeg` | `20131027025654Uh.jpeg` | **1.0000** | 0.00h |
| 63 | `20131027025654Uh.jpeg` | `20131027025654Uh.jpeg` | **1.0000** | 0.00h |
| 64 | `20141129222134Lh.jpeg` | `20141129222134Lh.jpeg` | **1.0000** | 0.00h |
| 65 | `20140605195854Bh.jpeg` | `20140605195854Bh.jpeg` | **1.0000** | 0.00h |
| 66 | `20150308081954Uh.jpeg` | `20150308081954Uh.jpeg` | **1.0000** | 0.00h |
| 67 | `20161205133714Th.jpeg` | `20161205133714Th.jpeg` | **1.0000** | 0.00h |
| 68 | `20160225173314Mh.jpeg` | `20160225173314Mh.jpeg` | **1.0000** | 0.00h |
| 69 | `20120318063154Uh.jpeg` | `20120318063154Uh.jpeg` | **1.0000** | 0.00h |
| 70 | `20161129022454Uh.jpeg` | `20161129022454Uh.jpeg` | **1.0000** | 0.00h |
| 71 | `20170831170750Bh.jpeg` | `20170831170750Bh.jpeg` | **1.0000** | 0.00h |
| 72 | `20200110073230Lh.jpeg` | `20200110073230Lh.jpeg` | **1.0000** | 0.00h |
| 73 | `20120225172914Mh.jpeg` | `20120225172914Mh.jpeg` | **1.0000** | 0.00h |
| 74 | `20110826074214Th.jpeg` | `20110826074214Th.jpeg` | **1.0000** | 0.00h |

### Detailed Pairwise Table (All 0-Day Validation Pairs)
| Validation Image | Nearest Training Image | Time Gap (h) | SSIM | Classification |
|:---|:---|:---:|:---:|:---|
| `20170425133730Ch.jpeg` | `20170425133730Ch.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20130425135954Bh.jpeg` | `20130425135954Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20160315133714Th.jpeg` | `20160315133714Th.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20160213153054Bh.jpeg` | `20160213153054Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20141210201814Mh.jpeg` | `20141210201814Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20170119031254Uh.jpeg` | `20170119031254Uh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20110410141654Bh.jpeg` | `20110410141654Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20170117175014Mh.jpeg` | `20170117175014Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20161231133734Ch.jpeg` | `20161231133734Ch.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20110327143554Bh.jpeg` | `20110327143554Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20170418013450Uh.jpeg` | `20170418013450Uh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20111027145954Bh.jpeg` | `20111027145954Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20160401170314Mh.jpeg` | `20160401170314Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20181101230010Mh.jpeg` | `20181101230010Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20111009110234Ch.jpeg` | `20111009110234Ch.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20150731082034Lh.jpeg` | `20150731082034Lh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20131025185414Mh.jpeg` | `20131025185414Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20140604195854Bh.jpeg` | `20140604195854Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20211225025850Uh.jpeg` | `20211225025850Uh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20110727063234Lh.jpeg` | `20110727063234Lh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20160501143154Bh.jpeg` | `20160501143154Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20150116082034Lh.jpeg` | `20150116082034Lh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20120814163114Mh.jpeg` | `20120814163114Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20190626025430Lh.jpeg` | `20190626025430Lh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20120913165314Mh.jpeg` | `20120913165314Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20170804140650Bh.jpeg` | `20170804140650Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20110128174814Mh.jpeg` | `20110128174814Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20150125172714Mh.jpeg` | `20150125172714Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20171008144250Bh.jpeg` | `20171008144250Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20170927143450Bh.jpeg` | `20170927143450Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20131006075314Th.jpeg` | `20131006075314Th.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20140312081114Th.jpeg` | `20140312081114Th.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20130903141654Bh.jpeg` | `20130903141654Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20150113105034Ch.jpeg` | `20150113105034Ch.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20110823120334Ch.jpeg` | `20110823120334Ch.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20171115222230Lh.jpeg` | `20171115222230Lh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20120113104834Ch.jpeg` | `20120113104834Ch.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20120723095014Th.jpeg` | `20120723095014Th.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20141016022554Uh.jpeg` | `20141016022554Uh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20121027165114Mh.jpeg` | `20121027165114Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20121129063154Uh.jpeg` | `20121129063154Uh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20121129063154Uh.jpeg` | `20121129063154Uh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20121027063134Lh.jpeg` | `20121027063134Lh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20171011170310Mh.jpeg` | `20171011170310Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20130925143254Bh.jpeg` | `20130925143254Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20211231223430Lh.jpeg` | `20211231223430Lh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20120603123034Ch.jpeg` | `20120603123034Ch.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20130811140254Bh.jpeg` | `20130811140254Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20130407063154Uh.jpeg` | `20130407063154Uh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20120205172414Mh.jpeg` | `20120205172414Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20131017185334Ch.jpeg` | `20131017185334Ch.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20110322005814Mh.jpeg` | `20110322005814Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20130107174854Bh.jpeg` | `20130107174854Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20130107174854Bh.jpeg` | `20130107174854Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20110815140454Bh.jpeg` | `20110815140454Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20190926165810Mh.jpeg` | `20190926165810Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20141216222634Lh.jpeg` | `20141216222634Lh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20150806082034Lh.jpeg` | `20150806082034Lh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20210817165030Mh.jpeg` | `20210817165030Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20170103030554Uh.jpeg` | `20170103030554Uh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20171022170710Mh.jpeg` | `20171022170710Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20131027025654Uh.jpeg` | `20131027025654Uh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20131027025654Uh.jpeg` | `20131027025654Uh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20141129222134Lh.jpeg` | `20141129222134Lh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20140605195854Bh.jpeg` | `20140605195854Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20150308081954Uh.jpeg` | `20150308081954Uh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20161205133714Th.jpeg` | `20161205133714Th.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20160225173314Mh.jpeg` | `20160225173314Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20120318063154Uh.jpeg` | `20120318063154Uh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20161129022454Uh.jpeg` | `20161129022454Uh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20170831170750Bh.jpeg` | `20170831170750Bh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20200110073230Lh.jpeg` | `20200110073230Lh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20120225172914Mh.jpeg` | `20120225172914Mh.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20110826074214Th.jpeg` | `20110826074214Th.jpeg` | 0.00h | 1.0000 | Near-Duplicate (Real Leakage) |
| `20210626060050Uh.jpeg` | `20210626133950Bh.jpeg` | 7.65h | 0.7003 | Clearly Different Observations (Not Leakage) |
| `20120223152054Bh.jpeg` | `20120223063154Uh.jpeg` | 8.82h | 0.6484 | Clearly Different Observations (Not Leakage) |
| `20151213023454Uh.jpeg` | `20151212171154Bh.jpeg` | 9.38h | 0.6268 | Clearly Different Observations (Not Leakage) |
| `20140615195854Bh.jpeg` | `20140615195814Mh.jpeg` | 0.01h | 0.6163 | Clearly Different Observations (Not Leakage) |
| `20110520063154Uh.jpeg` | `20110521063134Lh.jpeg` | 23.99h | 0.6137 | Clearly Different Observations (Not Leakage) |
| `20130304170614Mh.jpeg` | `20130305112734Ch.jpeg` | 18.36h | 0.5976 | Clearly Different Observations (Not Leakage) |
| `20141027224834Lh.jpeg` | `20141027025054Uh.jpeg` | 19.96h | 0.5888 | Clearly Different Observations (Not Leakage) |
| `20141027224834Lh.jpeg` | `20141027025054Uh.jpeg` | 19.96h | 0.5888 | Clearly Different Observations (Not Leakage) |
| `20131116185334Ch.jpeg` | `20131115185414Mh.jpeg` | 23.99h | 0.5855 | Clearly Different Observations (Not Leakage) |
| `20150527082014Th.jpeg` | `20150527161214Mh.jpeg` | 7.87h | 0.5609 | Clearly Different Observations (Not Leakage) |
| `20171025223130Lh.jpeg` | `20171025133730Ch.jpeg` | 8.90h | 0.5585 | Clearly Different Observations (Not Leakage) |
| `20171025223130Lh.jpeg` | `20171025133730Ch.jpeg` | 8.90h | 0.5585 | Clearly Different Observations (Not Leakage) |
| `20210329114430Ch.jpeg` | `20210329015250Uh.jpeg` | 9.86h | 0.5577 | Clearly Different Observations (Not Leakage) |
| `20171005133730Ch.jpeg` | `20171005144050Bh.jpeg` | 1.06h | 0.5412 | Clearly Different Observations (Not Leakage) |
| `20170702133730Ch.jpeg` | `20170701235630Lh.jpeg` | 13.68h | 0.5412 | Clearly Different Observations (Not Leakage) |
| `20130609173414Mh.jpeg` | `20130609123334Ch.jpeg` | 5.01h | 0.5400 | Clearly Different Observations (Not Leakage) |
| `20151118172314Mh.jpeg` | `20151119133734Ch.jpeg` | 20.24h | 0.5387 | Clearly Different Observations (Not Leakage) |
| `20120814140454Bh.jpeg` | `20120814163114Mh.jpeg` | 2.44h | 0.5065 | Clearly Different Observations (Not Leakage) |
| `20170503013050Uh.jpeg` | `20170502233130Lh.jpeg` | 1.99h | 0.4970 | Clearly Different Observations (Not Leakage) |
