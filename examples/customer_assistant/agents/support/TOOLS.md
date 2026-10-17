---
lookup_order:
  description: |-
    Fetch an order by its id

    Returns the order: its customer, items, total, card, status, carrier and tracking number, the
    promised and delivered dates, and every charge.
  input:
    order_id: {type: string, description: "The order id, such as A-1001", required: true}
---
