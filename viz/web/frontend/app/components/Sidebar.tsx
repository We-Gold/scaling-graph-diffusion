"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface SidebarProps {
  timestep: number;
  processType: "noise" | "denoise";
  isForwardProcess: boolean;
  noiseLevel: number;
  MAX_TIMESTEP: number;
}

export default function Sidebar({
  timestep,
  processType,
  isForwardProcess,
  noiseLevel,
  MAX_TIMESTEP,
}: SidebarProps) {
  const [activeTab, setActiveTab] = useState<"visualization" | "statistics">("visualization");

  return (
    <Card className="border-border/50 shadow-xl bg-card/50 backdrop-blur-sm h-full">
      <CardHeader>
        <CardTitle>Information</CardTitle>
        <CardDescription>View process data</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Tab Selector */}
        <div className="flex gap-2 p-1 bg-muted rounded-lg">
          <button
            onClick={() => setActiveTab("visualization")}
            className={`flex-1 px-3 py-2 text-sm font-medium rounded-md transition-colors ${
              activeTab === "visualization"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Visualization
          </button>
          <button
            onClick={() => setActiveTab("statistics")}
            className={`flex-1 px-3 py-2 text-sm font-medium rounded-md transition-colors ${
              activeTab === "statistics"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Statistics
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === "visualization" && (
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-semibold mb-2">Visualization Controls</h3>
              <p className="text-xs text-muted-foreground">
                Use the controls on the right to interact with the molecule visualization:
              </p>
              <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
                <li>• <span className="font-medium">Process Direction</span>: Switch between Denoise (Reverse) and Add Noise (Forward)</li>
                <li>• <span className="font-medium">Playback</span>: Play/Pause animation</li>
                <li>• <span className="font-medium">Timestep Slider</span>: Jump to specific timestep</li>
                <li>• <span className="font-medium">Speed</span>: Adjust animation speed</li>
              </ul>
            </div>

            <div className="pt-2 border-t border-border/30">
              <h3 className="text-sm font-semibold mb-2">Current State</h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Process:</span>
                  <span className="font-medium">{processType === "denoise" ? "Denoising" : "Adding Noise"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Direction:</span>
                  <span className="font-medium">{isForwardProcess ? "Forward" : "Reverse"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Timestep:</span>
                  <span className="font-medium font-mono">{timestep}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === "statistics" && (
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-semibold mb-2">Dataset Information</h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Process Type:</span>
                  <span className="font-medium capitalize">{processType}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Timesteps:</span>
                  <span className="font-medium">{MAX_TIMESTEP + 1}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Current Timestep:</span>
                  <span className="font-medium">{timestep}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Direction:</span>
                  <span className="font-medium">{isForwardProcess ? "Forward" : "Reverse"}</span>
                </div>
              </div>
            </div>

            <div className="pt-2 border-t border-border/30">
              <h3 className="text-sm font-semibold mb-3">Noise Analysis</h3>
              <div className="space-y-3">
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">Noise Level:</span>
                    <span className="font-mono font-medium">{(noiseLevel * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-muted rounded-full h-2">
                    <div 
                      className="h-full rounded-full transition-all duration-300"
                      style={{ 
                        width: `${noiseLevel * 100}%`,
                        background: 'linear-gradient(to right, rgb(34, 197, 94), rgb(234, 179, 8), rgb(239, 68, 68))'
                      }}
                    />
                  </div>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Clean:</span>
                  <span className="font-mono font-medium">{((1 - noiseLevel) * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Progress:</span>
                  <span className="font-mono font-medium">{((timestep / MAX_TIMESTEP) * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            <div className="pt-2 border-t border-border/30">
              <h3 className="text-sm font-semibold mb-2">Process Information</h3>
              <div className="space-y-2 text-xs text-muted-foreground">
                {processType === "denoise" ? (
                  <p>
                    <span className="font-medium text-foreground">Denoising Process:</span> Removes noise progressively to reveal the clean molecular structure. Starting from pure noise (t=0) to clean molecule (t={MAX_TIMESTEP}).
                  </p>
                ) : (
                  <p>
                    <span className="font-medium text-foreground">Noise Process:</span> Adds noise progressively to destroy the molecular structure. Starting from clean molecule (t=0) to pure noise (t={MAX_TIMESTEP}).
                  </p>
                )}
              </div>
            </div>

            <div className="pt-2 border-t border-border/30">
              <h3 className="text-sm font-semibold mb-2">Data Source</h3>
              <div className="space-y-1 text-xs text-muted-foreground">
                <p>
                  <span className="font-medium text-foreground">Path:</span> {processType}_process/raw/
                </p>
                <p>
                  <span className="font-medium text-foreground">Format:</span> NumPy .npz files
                </p>
                <p>
                  <span className="font-medium text-foreground">Total Files:</span> {MAX_TIMESTEP + 1} timesteps
                </p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
