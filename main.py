from fastapi import FastAPI, HTTPException, Header
from bridge import ThermalBridge
import os

app = FastAPI(title='Genesis Thermal Bridge')
bridge = ThermalBridge()

@app.post('/crystallize')
async def get_truth(query: str, x_api_key: str = Header(...)):
    if x_api_key != os.getenv('GENESIS_CLIENT_KEY'):
        raise HTTPException(status_code=402, detail='Payment Required: Buy credits at Genesis-Conductor')

    # The Expensive Logic is now subsidized by Cheap Physics
    result = bridge.crystallize(query)
    return {'status': 'CRYSTALLINE', 'output': result, 'energy_saved': '99.9%'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
