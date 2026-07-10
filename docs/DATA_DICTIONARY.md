# Data Dictionary

## Dataset Overview

This dataset contains customer transaction records used for analytics and reporting.

Maintained By: Data Engineering Team

---

# Column Details

## customer_id

- **Type:** Integer
- **Business Meaning:** Unique customer identifier
- **Example:** 1001
- **Null Handling:** Never null
- **Related KPI:** Customer Tracking
- **Updates:** Generated when customer registers

---

## name

- **Type:** String
- **Business Meaning:** Customer full name
- **Example:** Alice
- **Null Handling:** Missing values should be reviewed
- **Related KPI:** Customer Records
- **Updates:** Updated whenever profile changes

---

## email

- **Type:** String
- **Business Meaning:** Customer email used for communication
- **Example:** alice@example.com
- **Null Handling:** Missing values require follow-up
- **Related KPI:** Customer Engagement
- **Updates:** Editable by customer

---

## amount

- **Type:** Float
- **Business Meaning:** Revenue generated from a transaction
- **Example:** 150.75
- **Unit:** USD
- **Related KPI:** Monthly Revenue
- **Updates:** Recorded after successful payment

---

## status

- **Type:** String
- **Business Meaning:** Status of the transaction
- **Valid Values:** Completed, Pending
- **Related KPI:** Transaction Success Rate
- **Updates:** Updated after payment processing

---

# Column to KPI Mapping

## Monthly Revenue

- Formula: SUM(amount)
- Related Columns: amount
- Update Frequency: Daily

## Customer Tracking

- Formula: COUNT(customer_id)
- Related Columns: customer_id
- Update Frequency: Daily

## Customer Engagement

- Formula: COUNT(email)
- Related Columns: email
- Update Frequency: Weekly

## Transaction Success Rate

- Formula: Completed / Total Transactions
- Related Columns: status
- Update Frequency: Daily

## Revenue Per Customer

- Formula: SUM(amount) GROUP BY customer_id
- Related Columns: customer_id, amount
- Update Frequency: Monthly

---

# Ambiguous Columns

## status

- Original Meaning: Unclear
- Resolved Meaning: Transaction completion status
- Proposed Rename: transaction_status

## amount

- Original Meaning: Could refer to quantity or money
- Resolved Meaning: Transaction revenue in USD
- Proposed Rename: transaction_amount

---

# Column Relationships

## Revenue Per Customer

Related Columns:

- customer_id
- amount

Business Impact:

Identifies high-value customers.

---

## Transaction Status by Customer

Related Columns:

- customer_id
- status

Business Impact:

Measures customer transaction success.