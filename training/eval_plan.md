# Eval plan

## Gates
- reasoning accuracy
- tool success rate
- long-context retrieval accuracy
- code pass rate
- safety reject rate
- kernel reuse rate

## Discipline
- hold out every eval set
- never tune on eval prompts
- keep a frozen benchmark pack
- require a better checkpoint before promotion
- record scores with timestamps

## Failure policy
If eval regresses:
- do not promote the checkpoint
- inspect data issues first
- check routing and skill-bank drift second
- only then modify architecture
