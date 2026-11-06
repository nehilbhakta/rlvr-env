import type { ReactNode } from "react";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], display: "swap", variable: "--font-inter" });

export const metadata = {
  title: "RLVR Demo — Verifiable Rewards, Real Verifiers",
  description:
    "A 299K-param tiny transformer, a regex baseline and an oracle, scored live by the RLVR environment's own verifiers on 20 tasks.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body>
        <header className="site-bar">
          <div className="site-bar-inner">
            <span className="wordmark">
              <span className="logo-mark" aria-hidden="true">✓</span>
              RLVR Demo
            </span>
            <span className="tagline">Verifiable rewards · no model judgement, no partial credit</span>
          </div>
        </header>
        <main>{children}</main>
        <footer className="site-footer">
          Tiny model: decoder-only transformer, 299,136 params, trained from scratch on
          variants of these 20 tasks · Reward is 1.0 iff the verifier passes, else 0.0.
        </footer>
      </body>
    </html>
  );
}
