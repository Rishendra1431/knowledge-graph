import { NextResponse } from "next/server"

export const revalidate = 300

const KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

type KevVulnerability = {
  cveID: string
  vendorProject: string
  product: string
  vulnerabilityName: string
  dateAdded: string
  shortDescription: string
  dueDate: string
}

export async function GET() {
  try {
    const response = await fetch(KEV_URL, { cache: "no-store", headers: { accept: "application/json" } })
    if (!response.ok) throw new Error(`CISA feed returned ${response.status}`)
    const feed = (await response.json()) as { title?: string; catalogVersion?: string; vulnerabilities?: KevVulnerability[] }
    const vulnerabilities = (feed.vulnerabilities ?? []).slice(-8).reverse().map((item) => ({
      id: item.cveID,
      title: item.vulnerabilityName,
      vendor: item.vendorProject,
      product: item.product,
      added: item.dateAdded,
      due: item.dueDate,
      description: item.shortDescription,
    }))
    return NextResponse.json({ source: "CISA Known Exploited Vulnerabilities Catalog", catalogVersion: feed.catalogVersion, updatedAt: new Date().toISOString(), total: feed.vulnerabilities?.length ?? 0, vulnerabilities })
  } catch (error) {
    return NextResponse.json({ error: "Threat intelligence feed unavailable", detail: error instanceof Error ? error.message : "Unknown error" }, { status: 502 })
  }
}
