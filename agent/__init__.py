"""Orion agent package.

The provider-backed graph is intentionally not imported at package load time.
Use ``from agent.graph import graph`` when the full agent runtime is needed.
This keeps the database-backed support demo and voice module independently
startable.
"""
