    export const calculateRisk = (req, res) => {
  const {
    credit_score,
    monthly_income,
    account_balance,
    total_emi
  } = req.body;

  if (
    credit_score === undefined ||
    monthly_income === undefined ||
    account_balance === undefined ||
    total_emi === undefined
  ) {
    return res.status(400).json({
      error: "Missing required financial fields"
    });
  }

  // ---- Rule-based mock logic (temporary) ----
  let riskScore = 0.5;

  if (credit_score > 750) riskScore -= 0.15;
  else if (credit_score < 650) riskScore += 0.2;

  const emiRatio = total_emi / monthly_income;
  if (emiRatio > 0.4) riskScore += 0.2;
  else riskScore -= 0.1;

  riskScore = Math.min(Math.max(riskScore, 0), 1);

  const maxCredit =
    monthly_income * (credit_score > 700 ? 5 : 3);

  res.status(200).json({
    risk_score: Number(riskScore.toFixed(2)),
    risk_level:
      riskScore < 0.3 ? "LOW" :
      riskScore < 0.6 ? "MEDIUM" : "HIGH",
    credit_limit: {
      min: Math.floor(maxCredit * 0.4),
      max: Math.floor(maxCredit)
    }
  });
};
