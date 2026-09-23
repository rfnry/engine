1. Never quote a price or a delivery time from memory. Find the destination's row in
   knowledge/rates.csv with Grep and quote that row.
2. Shipping is free when the order total is at or above the row's free_over_eur; an empty
   free_over_eur means never free.
3. Days are business days from dispatch. Read knowledge/dispatch.txt to know when dispatch happens.
4. Outside the EU the customer pays customs; say so.
