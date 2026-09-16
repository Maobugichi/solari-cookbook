import "dotenv/config";
import { SolariClient } from "@solarisdk/sdk";
import { executeTool } from "./index";

async function main() {
  const solari = new SolariClient({ apiKey: process.env.SOLARI_API_KEY! });
  let sandbox: any;

  try {
    sandbox = await solari.sandboxes.create({ template: "base", timeoutMs: 5 * 60 * 1000 });
    await sandbox.connect();

    console.log("\n--- valid public URL (expect statusCode 200) ---");
    const ok = await executeTool(sandbox, "fetch_live_url", { url: "https://example.com" });
    console.log(JSON.stringify(ok, null, 2).slice(0, 400));
    console.assert(ok.statusCode === "200", "FAIL: expected 200");

    console.log("\n--- blocked internal target (expect throw) ---");
    try {
      await executeTool(sandbox, "fetch_live_url", { url: "http://169.254.169.254/latest/meta-data/" });
      console.error("SECURITY FAIL: internal target was not refused!");
    } catch (e) {
      console.log("Refused as expected:", (e as Error).message);
    }

    console.log("\n--- non-http(s) scheme (expect throw) ---");
    try {
      await executeTool(sandbox, "fetch_live_url", { url: "file:///etc/passwd" });
      console.error("SECURITY FAIL: file:// was not refused!");
    } catch (e) {
      console.log("Refused as expected:", (e as Error).message);
    }

    console.log("\n✅ fetch_live_url checks complete.");
  } finally {
    if (sandbox) await sandbox.kill().catch(() => {});
  }
}

main().catch((e) => { console.error(e); process.exitCode = 1; });