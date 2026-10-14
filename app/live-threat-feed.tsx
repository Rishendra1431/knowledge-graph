"use client"

import { useEffect, useState } from "react"

type Threat = { id: string; title: string; vendor: string; product: string; added: string; due: string; description: string }
type Feed = { source: string; catalogVersion?: string; updatedAt: string; total: number; vulnerabilities: Threat[]; error?: string; detail?: string }

export function LiveThreatFeed() {
  const [feed, setFeed] = useState<Feed | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    fetch("/api/threat-intel").then((response) => response.json()).then((data: Feed) => { if (active) setFeed(data) }).catch(() => { if (active) setFeed({ error: "Unable to reach live feed", source: "CISA KEV" } as Feed) }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [])

  return <section id="live-feed" className="panel feed-panel">
    <div className="section-heading"><div><p className="eyebrow">LIVE THREAT INTELLIGENCE</p><h2>Recently exploited vulnerabilities</h2></div><span className={`tag ${feed?.error ? "warning" : "green"}`}><i className="pulse" /> {loading ? "SYNCING" : feed?.error ? "DEGRADED" : "LIVE"}</span></div>
    {feed?.error ? <div className="feed-empty"><strong>Feed temporarily unavailable</strong><p>{feed.detail ?? "The serverless feed endpoint could not reach CISA."}</p></div> : <>
      <div className="feed-meta"><span>{feed?.source ?? "CISA Known Exploited Vulnerabilities Catalog"}</span><span>{feed ? `${feed.total.toLocaleString()} catalog entries` : "Loading catalog"}</span><span>{feed ? `Updated ${new Date(feed.updatedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` : ""}</span></div>
      <div className="threat-list">{(feed?.vulnerabilities ?? []).map((threat) => <article className="threat-row" key={threat.id}><div className="threat-id">{threat.id}<small>ADDED {threat.added}</small></div><div><strong>{threat.title}</strong><p>{threat.vendor} / {threat.product}</p></div><span className="due">DUE {threat.due}</span></article>)}{!loading && !feed?.vulnerabilities?.length && <div className="feed-empty">No recent vulnerabilities returned.</div>}</div>
    </>}
  </section>
}
