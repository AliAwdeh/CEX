import dotenv from "dotenv";
import fs from "node:fs";
import path from "node:path";
import pg from "pg";

const root = path.resolve(import.meta.dirname, "..");
const pipelineEnv = path.resolve(root, "..", ".env");
const localEnv = path.resolve(root, ".env");

if (fs.existsSync(pipelineEnv)) dotenv.config({ path: pipelineEnv, quiet: true });
if (fs.existsSync(localEnv)) dotenv.config({ path: localEnv, override: true, quiet: true });

function normalizeDatabaseUrl(value) {
  if (!value) return "postgresql://localhost:5432/cex_pipeline";
  return value.replace("postgresql+psycopg://", "postgresql://");
}

export const config = {
  port: Number(process.env.CX_DASHBOARD_API_PORT || 8090),
  statementTimeoutMs: Number(process.env.CX_DASHBOARD_STATEMENT_TIMEOUT_MS || 30000),
  databaseUrl: normalizeDatabaseUrl(process.env.CX_DASHBOARD_DATABASE_URL || process.env.DATABASE_URL),
  statsDatabaseUrl: normalizeDatabaseUrl(
    process.env.CX_DASHBOARD_STATS_DATABASE_URL || "postgresql://localhost:5432/cex_dashboard_stats"
  ),
  webhookSecret: process.env.CX_DASHBOARD_WEBHOOK_SECRET || ""
};

export const pool = new pg.Pool({
  connectionString: config.databaseUrl,
  max: 10,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 5000,
  application_name: "cx_react_readonly_dashboard"
});

export async function readonlyQuery(sql, params = []) {
  const normalized = sql.trim().toLowerCase();
  if (!normalized.startsWith("select") && !normalized.startsWith("with")) {
    throw new Error("Only read-only SELECT queries are allowed");
  }

  const client = await pool.connect();
  try {
    await client.query("BEGIN READ ONLY");
    await client.query("SELECT set_config('statement_timeout', $1, true)", [String(config.statementTimeoutMs)]);
    const result = await client.query(sql, params);
    await client.query("ROLLBACK");
    return result.rows;
  } catch (error) {
    try {
      await client.query("ROLLBACK");
    } catch {
      // The original query error is the one that matters.
    }
    throw error;
  } finally {
    client.release();
  }
}

export function asNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}
