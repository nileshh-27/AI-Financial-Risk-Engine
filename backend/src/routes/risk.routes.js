import { Router } from "express";
import { calculateRisk } from "../controllers/risk.controller.js";

const router = Router();

router.post("/score", calculateRisk);

export default router;
