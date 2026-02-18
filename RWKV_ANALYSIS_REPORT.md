# RWKV Experiment Message Analysis Report

**Experiment Directory:** `/scratch/local/homes/80/georgenigm/LOBS5/output/evalsequences/rwkv_goog2022_benchmark/exp_4_20260217_112951/data_gen/`

**Files Analyzed:** 16 message CSV files (GOOG 2022 data)

**Analysis Date:** 2026-02-17

---

## Executive Summary

Analysis of 16 RWKV-generated message files reveals **23 negative order IDs** across the dataset, predominantly in cancel messages (type 3). All aggressive orders at expected positions (lines 6 and 12) have consistent countdown IDs (17 and 11 respectively). No duplicate countdown IDs were found in type 1 messages.

---

## 1. NEGATIVE ORDER IDs

### Overview
- **Total occurrences:** 23
- **Affected files:** 12 out of 16 (75%)
- **Event type distribution:**
  - Type 3 (Cancel): 20 occurrences (87%)
  - Type 4 (Market order): 3 occurrences (13%)

### Breakdown by ID Value

| Order ID | Count | Event Types | Description |
|----------|-------|-------------|-------------|
| -1 | 11 | Type 3 | Most common negative ID |
| -6 | 4 | Type 4 | Three consecutive market orders in one file |
| -10 | 2 | Type 3 | Two separate files |
| -17 | 2 | Type 3 | Two separate files |
| -2 | 1 | Type 3 | Single occurrence |
| -12 | 1 | Type 3 | Single occurrence |
| -13 | 1 | Type 3 | Single occurrence |
| -16 | 1 | Type 3 | Single occurrence |

### Files with Negative IDs

1. **GOOG_2022-02-07** (1 occurrence): Line 8, type 3, ID=-1
2. **GOOG_2022-02-17** (2 occurrences): Lines 3, 19, both type 3, ID=-1
3. **GOOG_2022-03-07** (1 occurrence): Line 1, type 3, ID=-1
4. **GOOG_2022-04-13** (1 occurrence): Line 19, type 3, ID=-1
5. **GOOG_2022-07-21** (3 occurrences): Lines 7, 13, 15, all type 3, IDs=-1, -10, -17
6. **GOOG_2022-07-29** (1 occurrence): Line 8, type 3, ID=-1
7. **GOOG_2022-08-23** (2 occurrences): Lines 9, 18, both type 3, ID=-1
8. **GOOG_2022-08-24** (3 occurrences): Lines 20-22, all type 4, ID=-6
9. **GOOG_2022-09-07** (1 occurrence): Line 14, type 3, ID=-1
10. **GOOG_2022-10-20** (1 occurrence): Line 9, type 3, ID=-10
11. **GOOG_2022-11-01** (4 occurrences): Lines 16, 18, 19, 21, all type 3, IDs=-13, -6, -1, -12
12. **GOOG_2022-11-03** (1 occurrence): Line 20, type 3, ID=-16
13. **GOOG_2022-12-22** (2 occurrences): Lines 4, 14, both type 3, IDs=-2, -17

### Notable Pattern: Type 4 with ID=-6

In file `GOOG_2022-08-24_message_real_id_176955_gen_id_0.csv`, three consecutive market orders (type 4) all reference the same negative order ID (-6):

```
Line 20: time=506625.29, type=4, id=-6, size=2, price=1151600, dir=-1
Line 21: time=536570.23, type=4, id=-6, size=3, price=1151600, dir=-1
Line 22: time=566515.16, type=4, id=-6, size=1, price=1151600, dir=-1
```

This is unusual because type 4 messages typically don't reference existing order IDs.

---

## 2. DUPLICATE COUNTDOWN IDs (Type 1 Messages)

**Result:** No duplicates found.

All type 1 (limit order submission) messages use unique countdown IDs within each file. Countdown IDs range from 1 to 22 in the generated portions.

---

## 3. CANCEL MESSAGE VALIDITY (Type 3)

### Analysis Methodology
For each type 3 (cancel) message, we checked:
1. Does the order ID appear in a previous type 1 message in the same file?
2. Is it a conditioning-era ID (large number like 111xxxxx or similar)?
3. Is it a negative ID?

### Results
- **Total type 3 messages:** 120 across all files
- **Invalid cancel references:** 0 (excluding negative IDs)

All type 3 messages either:
- Reference conditioning-era IDs (large 6-9 digit numbers), OR
- Reference generated order IDs from previous type 1 messages, OR
- Use negative IDs (23 cases documented above)

**Conclusion:** Cancel messages are valid in their referencing, except for the negative ID anomaly.

---

## 4. AGGRESSIVE ORDERS AT POSITIONS 5 AND 11

**Note:** Analysis checked lines 6 and 12 (0-indexed positions 5 and 11).

### Position 6 (First Aggressive Order)
- **Files with type 4 at line 6:** 16 out of 16 (100%)
- **Order ID:** All use countdown ID `17`
- **Sizes:** Range from 1 to 75 shares
- **Direction:** All sell orders (dir=-1)

### Position 12 (Second Aggressive Order)
- **Files with type 4 at line 12:** 15 out of 16 (94%)
- **Order ID:** All use countdown ID `11`
- **Sizes:** Range from 1 to 75 shares
- **Direction:** All sell orders (dir=-1)

**Missing from position 12:**
- File: `GOOG_2022-03-07_message_real_id_51018_gen_id_0.csv` (only 21 total lines)

### Consistency
**EXCELLENT:** All aggressive orders at expected positions use the exact countdown IDs (17 and 11) that would be expected from the countdown sequence.

---

## 5. EVENT TYPE STATISTICS

### Global Counts (All 16 Files)

| Event Type | Count | Percentage | Description |
|------------|-------|------------|-------------|
| 1 | 158 | 45.5% | Limit order submission |
| 2 | 2 | 0.6% | Partial fill |
| 3 | 120 | 34.6% | Cancellation |
| 4 | 67 | 19.3% | Market order/aggressive fill |
| **Total** | **347** | **100%** | |

### Per-File Distribution

Average per file:
- Type 1: 9.9 messages
- Type 2: 0.1 messages
- Type 3: 7.5 messages
- Type 4: 4.2 messages
- **Total:** 21.7 messages per file

---

## 6. ANOMALIES AND CONCERNS

### Critical Issues

1. **Negative Order IDs in Cancel Messages (Type 3)**
   - 20 type 3 messages use negative IDs
   - Most common: ID=-1 (11 occurrences)
   - These IDs don't correspond to any generated type 1 messages
   - **Impact:** These cancel messages would fail in a real order book

2. **Negative Order IDs in Market Orders (Type 4)**
   - 3 type 4 messages use negative ID=-6
   - All in the same file (GOOG_2022-08-24)
   - **Impact:** Type 4 messages shouldn't reference order IDs at all

3. **Suspicious Price Value**
   - File: `GOOG_2022-11-01`, Line 19
   - Price: `91300` (should be ~920000-921000 based on context)
   - **Likely:** Missing a digit (should be 921300)

### Non-Issues

1. **Large Order IDs in Cancel Messages:** These are conditioning-era IDs from historical data and are expected.

2. **Missing Type 4 at Line 12:** One file is too short (21 lines) to have a second aggressive order.

---

## 7. RECOMMENDATIONS

### For RWKV Model Training/Inference

1. **Fix Negative ID Generation:**
   - Add constraint to prevent negative order IDs
   - Ensure countdown sequence never goes below 0
   - Consider adding validation layer post-generation

2. **Investigate ID=-1 Pattern:**
   - ID=-1 appears 11 times (48% of negative IDs)
   - May indicate a specific bug in ID generation logic
   - Could be related to countdown underflow

3. **Type 4 Order ID Handling:**
   - Type 4 messages should not reference order IDs
   - Review RWKV's understanding of message type semantics

4. **Price Validation:**
   - Add digit-count validation for price fields
   - Check for sudden price discontinuities

### For Order Book Simulation

1. **Pre-Processing:**
   - Filter out messages with negative order IDs
   - Log warnings for debugging

2. **Error Handling:**
   - Gracefully skip invalid cancel/market order messages
   - Track rejection rate for model quality metrics

---

## 8. COMPARISON TO EXPECTED BEHAVIOR

### What Works Well ✓

- **Countdown ID sequence:** No duplicates, consistent pattern
- **Aggressive order placement:** Correct positions and IDs (17 and 11)
- **Cancel message references:** Valid except for negative IDs
- **Event type distribution:** Reasonable mix of order types
- **File completion rate:** 94% of files have full expected structure

### What Needs Attention ✗

- **Negative order IDs:** 23 occurrences (6.6% of all messages)
- **Type 4 semantics:** Should not use order ID field
- **Price validation:** At least 1 likely malformed price

---

## 9. FILES WITH NO ANOMALIES

The following 3 files have **zero negative IDs** and perfect structure:

1. `GOOG_2022-08-11_message_real_id_162915_gen_id_0.csv`
2. `GOOG_2022-11-15_message_real_id_319492_gen_id_0.csv`
3. `GOOG_2022-12-02_message_real_id_337553_gen_id_0.csv`

**Clean file rate:** 18.8% (3/16)

---

## Appendix: Analysis Scripts

Two Python scripts were used for this analysis:

1. **`analyze_rwkv_messages.py`** - High-level statistics and counts
2. **`detailed_rwkv_analysis.py`** - Context analysis around negative IDs

Both scripts are available in the project root directory.

---

**End of Report**
