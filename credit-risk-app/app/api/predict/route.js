/**
 * POST /api/predict
 * Proxies request to the Python Flask backend at localhost:5001
 */
export async function POST(request) {
  try {
    const body = await request.json();

    const response = await fetch("http://localhost:5001/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const error = await response.json();
      return Response.json(
        { error: error.error || "Backend error" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return Response.json(data);
  } catch (error) {
    return Response.json(
      {
        error:
          "Could not connect to the ML server. Please ensure the Python backend is running on port 5001.",
      },
      { status: 503 }
    );
  }
}
