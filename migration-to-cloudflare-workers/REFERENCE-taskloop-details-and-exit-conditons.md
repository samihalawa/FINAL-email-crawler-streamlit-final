# Migration Requirements Reference

## Core Requirements

### Database (Supabase)
- Keep existing schema (see models in streamlit_app.py)
- No migrations needed
- Must maintain all relationships
- Zero data loss

### Frontend (Cloudflare Pages)
- Use Shadcn/UI components only
- Match Streamlit pages exactly:
  1. Manual Search
  2. Bulk Send
  3. View Leads
  4. Search Terms
  5. Email Templates
  6. Projects & Campaigns
  7. Knowledge Base
  8. AutoclientAI
  9. Automation Control
  10. Email Logs
  11. Settings
  12. Sent Campaigns

### API Layer (Cloudflare Workers)
Required endpoints for:
- Email operations
- Search operations
- Lead management
- Automation control
- Template management
- Project/campaign handling

### Compute Layer (HuggingFace Spaces)
Must handle:
- AI operations
- Heavy computations
- Email customization
- Search optimization

## Success Criteria

### Performance
- Page loads: < 3s
- API response: < 500ms
- Worker execution: < 10s

### Reliability
- 99% uptime
- 95% success rate
- 0% data loss

### Features
Must maintain exact parity with streamlit_app.py:
- All search capabilities
- All email functions
- All automation features
- All management tools

## Exit Conditions

### Required Working
1. Database
- All tables accessible
- All operations functional
- All relationships intact

2. Frontend
- All pages rendered
- All interactions working
- All exports functional

3. API
- All endpoints responding
- Error handling active
- Rate limiting working

4. Automation
- Scheduling working
- Tasks executing
- Results saving

### Quality Gates
1. All tests passing
2. Performance targets met
3. Security implemented
4. Documentation complete

## Task Loop Termination

### Must Complete
- All features working
- All integrations tested
- All automations running
- All docs updated

### Final Checks
- Component tests pass
- Integration tests pass
- Performance tests pass
- Security tests pass
