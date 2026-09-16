import "dotenv/config";
import { SolariClient } from "@solarisdk/sdk";
import { executeTool } from "./index"; // requires the export from Step 0

const FIXTURE_REPO_URL = "https://github.com/Maobugichi/solari-search-fixture";

async function main() {
  const solari = new SolariClient({ apiKey: process.env.SOLARI_API_KEY! });
  let sandbox: any;

  try {
    sandbox = await solari.sandboxes.create({ template: "base", timeoutMs: 5 * 60 * 1000 });
    await sandbox.connect();
    await sandbox.git.clone(FIXTURE_REPO_URL, { path: "/workspace/repo" });

    console.log("\n--- unscoped search (expect truncated: true, 200 matches) ---");
    const unscoped = await executeTool(sandbox, "search_files", { query: "NEEDLE_STRING" });
    console.log(JSON.stringify(unscoped, null, 2).slice(0, 500));
    console.assert(unscoped.truncated === true, "FAIL: expected truncation");
    console.assert((unscoped.matches as any[]).length === 200, "FAIL: expected 200 returned matches");

    console.log("\n--- scoped search (expect scopedTo: 'src', matchCount 252) ---");
    const scoped = await executeTool(sandbox, "search_files", { query: "NEEDLE_STRING", path: "src" });
    console.assert(scoped.scopedTo === "src", "FAIL: scopedTo missing/wrong");

    console.log("\n--- secret-exclusion search (expect matchCount 0) ---");
    const secretSearch = await executeTool(sandbox, "search_files", { query: "doNotFind123" });
    console.log(JSON.stringify(secretSearch));
    console.assert(secretSearch.matchCount === 0, "SECURITY FAIL: .env content leaked!");

    console.log("\n--- zero-match search (expect matchCount 0, no throw) ---");
    const noMatch = await executeTool(sandbox, "search_files", { query: "ZZZ_NOT_PRESENT_ZZZ" });
    console.assert(noMatch.matchCount === 0, "FAIL: expected zero matches");

    console.log("\n✅ All search_files assertions passed.");
  } finally {
    if (sandbox) {
      try {
        await sandbox.kill();
        console.log("✅ Sandbox terminated.");
      } catch (error) {
        console.error("⚠️ Failed to terminate sandbox:", error);
      }
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});