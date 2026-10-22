---
name: frontdesk
kind: decision
model: judge
description: Routes an incoming message
route:
  by: intent
  to: {billing: billing, technical: support}
  min_confidence: 0.6
---
