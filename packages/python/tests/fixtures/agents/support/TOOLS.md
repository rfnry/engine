---
lookup_order:
  description: |-
    Fetch an order by id

    Returns the order as JSON.
  input:
    order_id: {type: string, description: The order id, required: true}
    include_items: {type: boolean, description: Include line items, required: false}
now:
---
