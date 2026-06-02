"""
house-notes — minimal serverless CRUD for the viewer's pinned notes.

Backed by DynamoDB (table `house-notes`, partition key `id`). Each note is
stored as a JSON document under the `doc` attribute, which sidesteps DynamoDB's
float→Decimal handling and keeps the API a plain document store.

Exposed via an API Gateway HTTP API (the org SCP blocks public Lambda Function
URLs). This endpoint is effectively open — the viewer is a public page, so the
`x-notes-token` header is NOT a secret. It's a lightweight BOT FILTER: requests
that don't carry the marker (random scanners hitting the raw URL) get 403. Real
abuse/cost is capped by API Gateway throttling on the stage. CORS preflight
(OPTIONS) is short-circuited below before the filter.

Routes (method-based):
  GET                       -> list all notes            -> 200 [ {note}, ... ]
  POST | PUT  {note}        -> upsert a note (needs id)  -> 200 {note}
  DELETE      {id} or ?id=  -> delete a note             -> 200 {"deleted": id}
"""

import json
import os
import boto3

table = boto3.resource("dynamodb").Table(os.environ["TABLE"])
TOKEN = os.environ["TOKEN"]


def resp(code, body):
    return {
        "statusCode": code,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body),
    }


def handler(event, context):
    http = event.get("requestContext", {}).get("http", {})
    method = http.get("method", "GET")

    # CORS preflight: API Gateway injects the Access-Control-* headers; we just
    # need to answer without requiring the secret (browsers never send it on OPTIONS).
    if method == "OPTIONS":
        return {"statusCode": 204, "headers": {}, "body": ""}

    # Bot filter (not auth): the marker is public, so this only deflects naive
    # scanners. Determined callers can read it from the page — that's accepted.
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    if headers.get("x-notes-token") != TOKEN:
        return resp(403, {"error": "forbidden"})

    try:
        if method == "GET":
            items, kwargs = [], {}
            while True:
                page = table.scan(**kwargs)
                items += page.get("Items", [])
                if "LastEvaluatedKey" not in page:
                    break
                kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
            return resp(200, [json.loads(i["doc"]) for i in items])

        if method in ("POST", "PUT"):
            note = json.loads(event.get("body") or "{}")
            if not note.get("id"):
                return resp(400, {"error": "id required"})
            table.put_item(Item={"id": note["id"], "doc": json.dumps(note)})
            return resp(200, note)

        if method == "DELETE":
            nid = (event.get("queryStringParameters") or {}).get("id")
            if not nid and event.get("body"):
                nid = json.loads(event["body"]).get("id")
            if not nid:
                return resp(400, {"error": "id required"})
            table.delete_item(Key={"id": nid})
            return resp(200, {"deleted": nid})

        return resp(405, {"error": "method not allowed"})
    except Exception as e:  # noqa: BLE001 - surface errors to the client for a personal tool
        return resp(500, {"error": str(e)})
