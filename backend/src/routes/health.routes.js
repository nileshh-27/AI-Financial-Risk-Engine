import { Router } from "express";

const router = Router();

router.get("/", (req, res) => {
  res.status(200).json({
    status: "OK",
    service: "AI Financial Risk Engine",
    timestamp: new Date().toISOString()
  });
});

export default router;
