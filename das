[1mdiff --git a/src/agent/__pycache__/vetting_agent.cpython-314.pyc b/src/agent/__pycache__/vetting_agent.cpython-314.pyc[m
[1mindex 762d85c..68b77c1 100644[m
Binary files a/src/agent/__pycache__/vetting_agent.cpython-314.pyc and b/src/agent/__pycache__/vetting_agent.cpython-314.pyc differ
[1mdiff --git a/src/agent/vetting_agent.py b/src/agent/vetting_agent.py[m
[1mindex ad35245..daeac53 100644[m
[1m--- a/src/agent/vetting_agent.py[m
[1m+++ b/src/agent/vetting_agent.py[m
[36m@@ -185,7 +185,7 @@[m [mclass VettingAgent:[m
             """[m
 [m
             response = self.client.models.generate_content([m
[31m-                model="gemini-2.5-flash",[m
[32m+[m[32m                model="gemini-2.5-flash-lite",[m
                 contents=prompt[m
             )[m
 [m
[1mdiff --git a/src/scoring/__pycache__/ml_risk_scorer.cpython-313.pyc b/src/scoring/__pycache__/ml_risk_scorer.cpython-313.pyc[m
[1mindex 21db7f8..059edb9 100644[m
Binary files a/src/scoring/__pycache__/ml_risk_scorer.cpython-313.pyc and b/src/scoring/__pycache__/ml_risk_scorer.cpython-313.pyc differ
