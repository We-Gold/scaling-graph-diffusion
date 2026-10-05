from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import os
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from .timestep_service import TimestepService
from .molecule import graph_to_mol

logger = logging.getLogger("uvicorn")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Timestep Diffusion Visualization API")
    logger.info("Timestep service initialized")
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("API shutting down")

app = FastAPI(title="Molecule Diffusion Graph API", version="1.0.0", lifespan=lifespan)

# CORS for the Next.js frontend. Extra origins: VIZ_CORS_ORIGINS (comma separated).
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"] + [
    o.strip() for o in os.environ.get("VIZ_CORS_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample folder with noise_process/raw and denoise_process/raw. Default: viz/web/data/digress_moses_sample
DATA_DIR = Path(
    os.environ.get(
        "VIZ_DATA_DIR",
        Path(__file__).resolve().parents[3] / "data" / "digress_moses_sample",
    )
)
timestep_service = TimestepService(DATA_DIR)

@app.get("/")
async def root():
    return {
        "message": "Timestep Diffusion Visualization API",
        "version": "1.0.0",
        "endpoints": {
            "info": "/timestep/info",
            "svg": "/timestep/{process_type}/{timestep}/svg",
            "data": "/timestep/{process_type}/{timestep}/data"
        }
    }

# ============================================================================
# Timestep Diffusion Process Endpoints
# ============================================================================

@app.get("/timestep/info")
async def get_timestep_info():
    """Get information about available timestep data"""
    try:
        stats = timestep_service.get_statistics()
        return {
            "message": "Timestep diffusion process data",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting timestep info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/timestep/{process_type}/{timestep}/svg")
async def get_timestep_svg(
    process_type: str,
    timestep: int,
    width: int = Query(400, ge=100, le=1000, description="SVG width in pixels"),
    height: int = Query(400, ge=100, le=1000, description="SVG height in pixels")
):
    """
    Get molecule SVG at specific timestep for noise or denoise process.
    
    Args:
        process_type: 'noise' or 'denoise'
        timestep: Timestep number (0-500)
        width: SVG width
        height: SVG height
        
    Returns:
        SVG image of molecule at given timestep
    """
    try:
        # Validate process type
        if process_type not in ["noise", "denoise"]:
            raise HTTPException(
                status_code=400, 
                detail="process_type must be 'noise' or 'denoise'"
            )
        
        # Validate timestep range
        if not (0 <= timestep <= 500):
            raise HTTPException(
                status_code=400,
                detail="timestep must be between 0 and 500"
            )
        
        # Get molecule data at timestep
        atoms_data, bonds_data = timestep_service.get_molecule_at_timestep(
            process_type, 
            timestep
        )
        
        # Create RDKit molecule
        mol = graph_to_mol(atoms_data, bonds_data)
        
        if mol is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create molecule at timestep {timestep}"
            )
        
        # Generate SVG
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg_str = drawer.GetDrawingText()
        
        return Response(
            content=svg_str,
            media_type="image/svg+xml",
            headers={
                "Content-Disposition": f"inline; filename={process_type}_t{timestep}.svg"
            }
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating SVG for {process_type} at timestep {timestep}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/timestep/{process_type}/{timestep}/data")
async def get_timestep_data(process_type: str, timestep: int):
    """
    Get raw molecule data at specific timestep.
    
    Args:
        process_type: 'noise' or 'denoise'
        timestep: Timestep number (0-500)
        
    Returns:
        JSON with atoms and bonds data
    """
    try:
        if process_type not in ["noise", "denoise"]:
            raise HTTPException(
                status_code=400,
                detail="process_type must be 'noise' or 'denoise'"
            )
        
        atoms_data, bonds_data = timestep_service.get_molecule_at_timestep(
            process_type,
            timestep
        )
        
        return {
            "process_type": process_type,
            "timestep": timestep,
            "atoms": atoms_data,
            "bonds": bonds_data,
            "num_atoms": len(atoms_data),
            "num_bonds": len(bonds_data)
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting data for {process_type} at timestep {timestep}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)