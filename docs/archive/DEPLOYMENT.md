# Deployment

This prototype is meant to be close to deployable.

## Run locally

```bash
python -m kernelweave.cli init ./store
python -m kernelweave.cli add-sample ./store
python -m kernelweave.cli list ./store
python -m kernelweave.cli plan ./store "make a safe shell command"
```

## Notes
- The store is file-based and easy to inspect.
- Kernels are JSON.
- Traces are JSON.
- No external service is required for the prototype.
