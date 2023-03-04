from agq.core.scenario import Scenario


def test_scenario(test_values):
    s_pa = Scenario(values=test_values)
    s_pa.run()
    s_pa.summary()
