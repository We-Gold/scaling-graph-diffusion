"use client";

import { useState, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { Play, Pause, RotateCcw, Loader2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ThemeToggle } from "@/components/theme-toggle";

const MIN_TIMESTEP = 0;
const MAX_TIMESTEP = 500;
const STEP_SIZE = 5;
const DEFAULT_INTERVAL = 100;

// Auto-detect backend URL: use same host as frontend
const getBackendURL = () => {
  // NEXT_PUBLIC_API_URL is inlined at build time (set it before `npm run build` or `npm run dev`).
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, '');
  }
  if (typeof window !== 'undefined') {
    // Client-side: use same hostname as frontend
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    return `${protocol}//${hostname}:8000`;
  }
  // Server-side: use localhost
  return 'http://localhost:8000';
};

const fetchTimestepSVG = async (processType: string, timestep: number): Promise<string> => {
  const backendURL = getBackendURL();
  const response = await fetch(`${backendURL}/timestep/${processType}/${timestep}/svg`);
  if (!response.ok) {
    throw new Error("Failed to fetch timestep SVG");
  }
  return response.text();
};

export default function MoleculeVisualization() {
  const [timestep, setTimestep] = useState(MAX_TIMESTEP); 
  const [isPlaying, setIsPlaying] = useState(false);
  const [interval, setInterval] = useState(DEFAULT_INTERVAL);
  const [previousData, setPreviousData] = useState<string>("");
  const [isForwardProcess, setIsForwardProcess] = useState(false); // false = Reverse (default), true = Forward
  const [processType, setProcessType] = useState<"noise" | "denoise">("denoise"); // Toggle between noise and denoise
  const intervalRef = useRef<number | null>(null);

  // Calculate noise level based on timestep (0 = pure noise, 500 = clean)
  const noiseLevel = 1 - (timestep / MAX_TIMESTEP); // 0.0 = clean, 1.0 = full noise

  // Fetch molecule SVG at current timestep
  const { data, isLoading, error } = useQuery({
    queryKey: ["timestep", processType, timestep],
    queryFn: () => fetchTimestepSVG(processType, timestep),
  });

  useEffect(() => {
    if (data && !isLoading) {
      setPreviousData(data);
    }
  }, [data, isLoading]);

  // Handle play functionality - animate TIMESTEP
  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = window.setInterval(() => {
        setTimestep((prevStep) => {
          let nextStep;
          
          if (isForwardProcess) {
            // Forward: 500 → 0 (clean → noisy)
            nextStep = prevStep - STEP_SIZE;
            if (nextStep < MIN_TIMESTEP) {
              setIsPlaying(false);
              return MIN_TIMESTEP;
            }
          } else {
            // Reverse: 0 → 500 (noisy → clean)
            nextStep = prevStep + STEP_SIZE;
            if (nextStep > MAX_TIMESTEP) {
              setIsPlaying(false);
              return MAX_TIMESTEP;
            }
          }
          
          return nextStep;
        });
      }, interval);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, interval, isForwardProcess]);

  const togglePlay = () => {
    setIsPlaying(!isPlaying);
  };

  const resetToStart = () => {
    setIsPlaying(false);
    // Reset to start of current process direction
    if (isForwardProcess) {
      setTimestep(MAX_TIMESTEP); // Forward starts from clean
    } else {
      setTimestep(MIN_TIMESTEP); // Reverse starts from noise
    }
  };

  const displayData = data || previousData;
  const showLoading = isLoading && !previousData;

  return (
    <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <Card className="border-border/50 shadow-2xl bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-2xl">Diffusion Molecule</CardTitle>
                    <CardDescription>
                      At timestep t={timestep} • Noise: {(noiseLevel * 100).toFixed(0)}%
                    </CardDescription>
                  </div>
                  {isLoading && (
                    <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                  )}
                </div>
              </CardHeader>
              <CardContent>
                <div className="relative aspect-square w-full rounded-lg overflow-hidden bg-muted/10 border border-border/30">
                  {error ? (
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-destructive">
                      <AlertCircle className="h-16 w-16" />
                      <div className="text-center">
                        <p className="font-semibold text-lg">Connection Error</p>
                        <p className="text-sm text-muted-foreground mt-1">
                          Unable to connect to backend server
                        </p>
                        <p className="text-xs text-muted-foreground mt-2">
                          Please ensure the backend is running on port 8000
                        </p>
                      </div>
                    </div>
                  ) : showLoading ? (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <Skeleton className="w-full h-full" />
                    </div>
                  ) : (
                    <div
                      className="absolute inset-0 flex items-center justify-center transition-all duration-300"
                      style={{ 
                        transform: 'scale(1.8)'
                      }}
                      dangerouslySetInnerHTML={{ __html: displayData || "" }}
                    />
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="space-y-6">
                            {/* Process Direction Toggle */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">Process Direction</label>
                    <Badge variant={isForwardProcess ? "default" : "secondary"}>
                      {isForwardProcess ? "Forward" : "Reverse"}
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      onClick={() => {
                        setIsForwardProcess(false);
                        setProcessType("denoise");
                        setIsPlaying(false);
                      }}
                      variant={!isForwardProcess ? "default" : "outline"}
                      className="w-full"
                    >
                      ← Reverse
                      <span className="ml-2 text-xs opacity-70">(Denoise)</span>
                    </Button>
                    <Button
                      onClick={() => {
                        setIsForwardProcess(true);
                        setProcessType("noise");
                        setIsPlaying(false);
                      }}
                      variant={isForwardProcess ? "default" : "outline"}
                      className="w-full"
                    >
                      Forward →
                      <span className="ml-2 text-xs opacity-70">(Add Noise)</span>
                    </Button>
                  </div>
                </div>


            <Card className="border-border/50 shadow-xl bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle>Playback Controls</CardTitle>
                <CardDescription>Control the diffusion animation</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex gap-2">
                  <Button
                    onClick={togglePlay}
                    className="flex-1"
                    size="lg"
                    variant={isPlaying ? "destructive" : "default"}
                  >
                    {isPlaying ? (
                      <>
                        <Pause className="mr-2 h-5 w-5" />
                        Pause
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-5 w-5" />
                        Play
                      </>
                    )}
                  </Button>
                  <Button
                    onClick={resetToStart}
                    size="lg"
                    variant="outline"
                  >
                    <RotateCcw className="h-5 w-5" />
                  </Button>
                </div>

                {/* Timestep Slider */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">
                      Diffusion Timestep
                    </label>
                    <Badge variant="secondary">t={timestep}</Badge>
                  </div>
                  <Slider
                    value={[timestep]}
                    onValueChange={(value) => setTimestep(value[0])}
                    min={MIN_TIMESTEP}
                    max={MAX_TIMESTEP}
                    step={10}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">
                    {!isForwardProcess 
                      ? "t=0 is fully noisy, t=500 is clean molecule (mode Denoising)"
                      : "t=0 is clean molecule, t=500 is fully noisy (mode Noising)"}
                  </p>
                  <div className="pt-2 space-y-1">
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
                </div>

                <div className="space-y-3">
                  <label className="text-sm font-medium">
                    Animation Speed
                  </label>
                  <select
                    value={interval}
                    onChange={(e) => setInterval(Number(e.target.value))}
                    disabled={isPlaying}
                    className="w-full h-10 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <option value={500}>Slow (500ms)</option>
                    <option value={250}>Normal (250ms)</option>
                    <option value={100}>Fast (100ms)</option>
                  </select>
                </div>
              </CardContent>
            </Card>

            <Card className="border-border/50 shadow-xl bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle>About Diffusion Processes</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-sm text-muted-foreground">
                <div>
                  <p className="font-medium text-foreground mb-2">← Reverse Process (Denoising)</p>
                  <p>
                    Gradually removes noise to generate clear molecular structures. 
                    This is how diffusion models create new molecules.
                  </p>
                </div>
                
                <div>
                  <p className="font-medium text-foreground mb-2">Forward Process (Adding Noise) →</p>
                  <p>
                    Progressively adds noise to destroy structure. 
                    This process trains the model to understand noise patterns.
                  </p>
                </div>

                <div className="pt-2 border-t border-border/30 space-y-2">
                  <div className="flex items-start gap-2">
                    <div className="w-2 h-2 rounded-full bg-blue-500 mt-1.5" />
                    <div>
                      <span className="font-medium text-foreground">Noisy:</span> Random, unstructured molecular graphs
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-2 h-2 rounded-full bg-pink-500 mt-1.5" />
                    <div>
                      <span className="font-medium text-foreground">Clear:</span> Well-defined, valid molecular structures
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
  );
}
