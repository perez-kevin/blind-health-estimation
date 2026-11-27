# Overlap check: our 50 NHANES questions vs OpenEstimate (Renda et al., ICLR 2026)

Source: github.com/alanarenda/openestimate, data/variables/nhanes_variables.json (74 statistics, NHANES 2017-2018, adults 18+), read 2026-11-25.

OpenEstimate targets: means of waist, weight, total cholesterol, blood lead/cadmium/mercury/manganese; probabilities of
self-reported diabetes, cancer, heart attack, depression medication, insulin use, health insurance, difficulty concentrating.
Conditions are 1-3 of the same variables (quantile bins or yes/no). No age, sex, ethnicity, nativity, sleep, diet, PHQ-9,
HbA1c, BMI category, self-perceived weight or work-schedule conditions.

Result: 0 of 50 questions ask the same statistic (same target and same subgroup).
11 questions share at least one variable, always in a different role or subgroup:

| Our Q | Shared variable | Ours | Closest OpenEstimate statistic |
|---|---|---|---|
| Q13 | blood mercury | ratio of geometric means by years in US, foreign-born Hispanic adults | mean mercury given cancer / heart attack / cholesterol, weight or cadmium bins |
| Q04 | diagnosed diabetes | % excellent/very good health among undiagnosed HbA1c >= 6.5% | P(diabetes) given weight, cholesterol, lead bins |
| Q05, Q18, Q44, Q45 | waist | self-perceived weight x waist; waist-to-hip ratio | mean waist given insulin / diabetes / insurance + heart attack |
| Q06, Q26, Q40 | weight | age at heaviest weight; self-reported vs measured BMI; greatest-ever minus current weight | mean weight given cadmium / waist / cholesterol bins |
| Q33, Q46 | depression | PHQ-9 score (not medication) with sleep or HbA1c | P(depression medication) given heart attack / cancer / insurance |
