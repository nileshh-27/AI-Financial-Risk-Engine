import dotenv from "dotenv";
import app from "./app.js";

import healthRoutes from "./routes/health.routes.js";
import riskRoutes from "./routes/risk.routes.js";

dotenv.config();

const PORT = process.env.PORT || 6969;

app.use("/health", healthRoutes);
app.use("/risk", riskRoutes);

app.listen(PORT, () => {
  console.log(`Backend running on port ${PORT}`);
});
