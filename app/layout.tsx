import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "ThreatGraph | Knowledge graph threat detection",
  description: "Explainable knowledge graph-based cyber threat detection results.",
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>
}
