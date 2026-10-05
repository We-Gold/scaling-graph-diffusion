"use client";

import { useState } from "react";
import MoleculeVisualization from "./MoleculeVisualization";
import Statistics from "./Statistics";
import { ThemeToggle } from "@/components/theme-toggle";

export default function MainView() {
  const [activeTab, setActiveTab] = useState<"visualization" | "statistics">("visualization");

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      {/* Tab Header */}
      <header className="border-b border-border/40 backdrop-blur-xl bg-background/80 sticky top-0 z-50">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between py-6">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-foreground to-foreground/60 bg-clip-text text-transparent">
                Graph Diffusion Visualization
              </h1>
              <p className="text-muted-foreground mt-1">
                Explore molecules at different diffusion timesteps
              </p>
            </div>
            <ThemeToggle />
          </div>

          {/* Tab Navigation */}
          <div className="flex gap-2 pb-2">
            <button
              onClick={() => setActiveTab("visualization")}
              className={`px-6 py-3 text-sm font-medium rounded-t-lg transition-all ${
                activeTab === "visualization"
                  ? "bg-background text-foreground border-b-2 border-primary"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
              }`}
            >
              Visualization
            </button>
            <button
              onClick={() => setActiveTab("statistics")}
              className={`px-6 py-3 text-sm font-medium rounded-t-lg transition-all ${
                activeTab === "statistics"
                  ? "bg-background text-foreground border-b-2 border-primary"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
              }`}
            >
              Statistics
            </button>
          </div>
        </div>
      </header>

      {/* Tab Content */}
      <div className="relative">
        {activeTab === "visualization" && <MoleculeVisualization />}
        {activeTab === "statistics" && <Statistics />}
      </div>
    </div>
  );
}
