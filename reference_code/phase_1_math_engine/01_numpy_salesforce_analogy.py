# Source: AI_Learning_Cursor lines 370-386
# Original transcript phase: None - None
# Nearest header: #### 🔗 Salesforce Analogy
# Title: 🔗 Salesforce Analogy
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

# ❌ Apex anti-pattern:
# for(Account acc : accounts) {
#     insert new Contact(AccountId=acc.Id);  // DML in loop!
# }

# ❌ Python/AI anti-pattern:
result = []
for i in range(1000000):
    result.append(a[i] * b[i])  # Loop multiplication!

# ✅ Salesforce fix: Batch DML
# insert contactList;  // Single bulk operation

# ✅ Python/AI fix: Vectorized operation
result = a * b  # NumPy multiplies ALL elements at once
