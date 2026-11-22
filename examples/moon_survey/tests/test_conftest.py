async def test_the_pool_sees_its_own_schema(pool):
    path = await pool.fetchval("show search_path")
    assert path.startswith("moon_test_")
