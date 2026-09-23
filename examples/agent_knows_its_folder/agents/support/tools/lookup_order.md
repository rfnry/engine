---
name: lookup_order
description: Fetch an order by id
input:
  order_id: {type: string, description: The order id, e.g. A-1, required: true}
---
Returns the order with its customer, amount, charges, card, and promised and delivered dates.
