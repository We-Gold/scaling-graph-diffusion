"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function Statistics() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h2 className="text-3xl font-bold mb-2">Dataset Statistics</h2>
          <p className="text-muted-foreground">
            Comprehensive analysis of the graph diffusion dataset including timestep information, file sizes, and data structure
          </p>
        </div>

        {/* Quick Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Total Timesteps
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">501</div>
              <p className="text-xs text-muted-foreground mt-1">t=0 to t=500</p>
            </CardContent>
          </Card>

          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Process Types
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">2</div>
              <p className="text-xs text-muted-foreground mt-1">Noise & Denoise</p>
            </CardContent>
          </Card>

          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Total Files
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">1,002</div>
              <p className="text-xs text-muted-foreground mt-1">501 × 2 processes</p>
            </CardContent>
          </Card>

          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Data Format
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">.npz</div>
              <p className="text-xs text-muted-foreground mt-1">NumPy compressed</p>
            </CardContent>
          </Card>
        </div>

        {/* Process Comparison Table */}
        <Card className="border-border/50 shadow-xl bg-card/50 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-xl">Process Comparison</CardTitle>
            <CardDescription>Detailed comparison between Noise and Denoise processes</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b-2 border-border/30">
                    <th className="text-left py-3 px-4 font-semibold">Property</th>
                    <th className="text-left py-3 px-4 font-semibold">
                      <Badge variant="secondary" className="font-normal">Denoise Process</Badge>
                    </th>
                    <th className="text-left py-3 px-4 font-semibold">
                      <Badge variant="default" className="font-normal">Noise Process</Badge>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border/20">
                    <td className="py-3 px-4 font-medium">Direction</td>
                    <td className="py-3 px-4 text-muted-foreground">← Reverse (Noisy → Clean)</td>
                    <td className="py-3 px-4 text-muted-foreground">→ Forward (Clean → Noisy)</td>
                  </tr>
                  <tr className="border-b border-border/20 bg-muted/20">
                    <td className="py-3 px-4 font-medium">Start State (t=0)</td>
                    <td className="py-3 px-4 text-muted-foreground">Fully Noisy Structure</td>
                    <td className="py-3 px-4 text-muted-foreground">Clean Original Molecule</td>
                  </tr>
                  <tr className="border-b border-border/20">
                    <td className="py-3 px-4 font-medium">End State (t=500)</td>
                    <td className="py-3 px-4 text-muted-foreground">Generated Molecule (not the forward one)</td>
                    <td className="py-3 px-4 text-muted-foreground">Fully Noisy Structure</td>
                  </tr>
                  <tr className="border-b border-border/20 bg-muted/20">
                    <td className="py-3 px-4 font-medium">Purpose</td>
                    <td className="py-3 px-4 text-muted-foreground">Molecule Generation</td>
                    <td className="py-3 px-4 text-muted-foreground">Model Training</td>
                  </tr>
                  <tr className="border-b border-border/20">
                    <td className="py-3 px-4 font-medium">Total Files</td>
                    <td className="py-3 px-4 text-muted-foreground">501 timesteps</td>
                    <td className="py-3 px-4 text-muted-foreground">501 timesteps</td>
                  </tr>
                  <tr className="border-b border-border/20 bg-muted/20">
                    <td className="py-3 px-4 font-medium">File Size Range</td>
                    <td className="py-3 px-4 text-muted-foreground">3.3 KB - 6 KB</td>
                    <td className="py-3 px-4 text-muted-foreground">6 KB - 14.8 KB</td>
                  </tr>
                  <tr className="border-b border-border/20">
                    <td className="py-3 px-4 font-medium">Avg File Size</td>
                    <td className="py-3 px-4 text-muted-foreground">~4.6 KB</td>
                    <td className="py-3 px-4 text-muted-foreground">~10.4 KB</td>
                  </tr>
                  <tr className="border-b border-border/20 bg-muted/20">
                    <td className="py-3 px-4 font-medium">Data Path</td>
                    <td className="py-3 px-4">
                      <code className="text-xs bg-muted px-2 py-1 rounded font-mono">
                        denoise_process/raw/
                      </code>
                    </td>
                    <td className="py-3 px-4">
                      <code className="text-xs bg-muted px-2 py-1 rounded font-mono">
                        noise_process/raw/
                      </code>
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium">File Naming</td>
                    <td className="py-3 px-4">
                      <code className="text-xs bg-muted px-2 py-1 rounded font-mono">
                        step_00-500_denoised.npz
                      </code>
                    </td>
                    <td className="py-3 px-4">
                      <code className="text-xs bg-muted px-2 py-1 rounded font-mono">
                        step_00-500_noisy.npz
                      </code>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Data Structure */}
        <Card className="border-border/50 shadow-xl bg-card/50 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-xl">NPZ File Structure</CardTitle>
            <CardDescription>Graph representation format in NumPy compressed files</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Badge variant="outline">nodes</Badge>
                <span>Array</span>
              </h3>
              <div className="bg-muted/30 rounded-lg p-4 space-y-2">
                <div className="flex justify-between items-start">
                  <span className="font-medium">Format:</span>
                  <code className="text-sm bg-background px-2 py-1 rounded">1D integer array</code>
                </div>
                <div className="flex justify-between items-start">
                  <span className="font-medium">Content:</span>
                  <span className="text-sm text-muted-foreground">Atom type indices</span>
                </div>
                <div className="flex justify-between items-start">
                  <span className="font-medium">Padding:</span>
                  <span className="flex items-center gap-2">
                    <code className="text-sm bg-background px-2 py-1 rounded">-1</code>
                    <span className="text-xs text-muted-foreground">(indicates padding)</span>
                  </span>
                </div>
                <div className="pt-2 border-t border-border/30">
                  <span className="font-medium">Atom Types:</span>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'].map((atom) => (
                      <Badge key={atom} variant="secondary" className="font-mono">
                        {atom}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Badge variant="outline">edges</Badge>
                <span>Matrix</span>
              </h3>
              <div className="bg-muted/30 rounded-lg p-4 space-y-2">
                <div className="flex justify-between items-start">
                  <span className="font-medium">Format:</span>
                  <code className="text-sm bg-background px-2 py-1 rounded">2D adjacency matrix</code>
                </div>
                <div className="flex justify-between items-start">
                  <span className="font-medium">Content:</span>
                  <span className="text-sm text-muted-foreground">Bond types between atoms</span>
                </div>
                <div className="pt-2 border-t border-border/30">
                  <span className="font-medium">Bond Types:</span>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mt-2">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="font-mono">0</Badge>
                      <span className="text-sm text-muted-foreground">None</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="font-mono">1</Badge>
                      <span className="text-sm text-muted-foreground">Single</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="font-mono">2</Badge>
                      <span className="text-sm text-muted-foreground">Double</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="font-mono">3</Badge>
                      <span className="text-sm text-muted-foreground">Triple</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="font-mono">4</Badge>
                      <span className="text-sm text-muted-foreground">Aromatic</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Timestep & Storage Info */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="border-border/50 shadow-xl bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-xl">Timestep Information</CardTitle>
              <CardDescription>Coverage and distribution</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center py-2">
                <span className="font-medium">Total Steps:</span>
                <Badge variant="secondary" className="text-lg px-4">501</Badge>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="font-medium">Range:</span>
                <code className="bg-muted px-3 py-1 rounded font-mono">t=0 → t=500</code>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="font-medium">Step Size:</span>
                <Badge variant="outline" className="font-mono">1</Badge>
              </div>
              <div className="pt-3 border-t border-border/30">
                <p className="text-sm text-muted-foreground">
                  Each timestep represents a distinct diffusion state, allowing for fine-grained analysis 
                  of the molecule transformation process.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border/50 shadow-xl bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-xl">Storage Information</CardTitle>
              <CardDescription>Total dataset size</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center py-2">
                <span className="font-medium">Denoise Files:</span>
                <span className="text-muted-foreground">~2.3 MB</span>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="font-medium">Noise Files:</span>
                <span className="text-muted-foreground">~5.2 MB</span>
              </div>
              <div className="flex justify-between items-center py-2 border-t border-border/30 pt-3">
                <span className="font-medium">Total Dataset:</span>
                <Badge variant="default" className="text-lg px-4">~7.5 MB</Badge>
              </div>
              <div className="pt-3 border-t border-border/30">
                <p className="text-sm text-muted-foreground">
                  Compact NPZ format enables efficient storage and fast loading of molecular graph data 
                  for visualization and analysis.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
