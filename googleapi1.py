async def do_manual_search(request: Request, db: Session = Depends(get_db)):
    data = await request.form()
    terms = data.get("terms", "").split(',')
    num_results = int(data.get("num_results", 10))
    language = data.get("language", "ES")
    
    all_results = []
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }

    data = await request.form()
    terms = data.get("terms", "").split(',')
    num_results = int(data.get("num_results", 10))
    language = data.get("language", "ES")
    
    all_results = []
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }

    data = await request.form()
    terms = data.get("terms", "").split(',')
    num_results = int(data.get("num_results", 10))
    language = data.get("language", "ES")
    
    all_results = []
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }

    data = await request.form()
    terms = data.get("terms", "").split(',')
    num_results = int(data.get("num_results", 10))
    language = data.get("language", "ES")
    
    all_results = []
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }

    data = await request.form()
    terms = data.get("terms", "").split(',')
    num_results = int(data.get("num_results", 10))
    language = data.get("language", "ES")
    
    all_results = []
    total_emails = 0
    
    for term in terms:
    all_results = []
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                @app.post("/manual_search")
                async def do_manual_search(request: Request, db: Session = Depends(get_db)):
                    data = await request.json()
                    terms = data.get("terms", [])
                    num_results = data.get("num_results", 10)
                    language = data.get("language", "ES")
                
                    all_results = []
                    total_emails = 0
                
                    for term in terms:
                        try:
                            search_results = search_google(term.strip(), db)
                            emails, sources = extract_emails(search_results)
                
                            for email in emails:
                                for source in sources:
                                    save_to_db(db, email, source, term.strip())
                
                            all_results.append({
                                "term": term.strip(),
                                "emails_found": list(emails),
                                "sources": sources,
                                "total_results": len(emails)
                            })
                
                            total_emails += len(emails)
                
                        except Exception as e:
                            print(f"Error searching term {term.strip()}: {str(e)}")
                            continue
                
                    return {
                        "total_emails": total_emails,
                        "results": all_results
                    }
                
                        except Exception as e:
                            print(f"Error searching term {term}: {str(e)}")
                            continue
                
                    return {
                        "total_emails": total_emails,
                        "results": all_results
                    }
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
    data = await request.form()
    terms = data.get("terms", "").split(',')
    num_results = int(data.get("num_results", 10))
    language = data.get("language", "ES")
    
    all_results = []
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }

    data = await request.form()
    terms = data.get("terms", "").split(',')
    num_results = int(data.get("num_results", 10))
    language = data.get("language", "ES")
    
    all_results = []
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }

    data = await request.form()
    terms = data.get("terms", "").split(',')
    num_results = int(data.get("num_results", 10))
    language = data.get("language", "ES")
    
    all_results = []
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }

    data = await request.form()
    terms = data.get("terms", "").split(',')
    num_results = int(data.get("num_results", 10))
    language = data.get("language", "ES")
    
    all_results = []
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }

    data = await request.form()
    terms = data.get("terms", "").split(',')
    num_results = int(data.get("num_results", 10))
    language = data.get("language", "ES")
    
    all_results = []
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }

    data = await request.form()
    terms = data.get("terms", "").split(',')
    num_results = int(data.get("num_results", 10))
    language = data.get("language", "ES")
    
    all_results = []
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term.strip(), db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term.strip())
            
            all_results.append({
                "term": term.strip(),
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term.strip()}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }

    data = await request.form()
    terms = data.get("terms", "").split(',')
    num_results = int(data.get("num_results", 10))
    language = data.get("language", "ES")

import os
import json
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, BigInteger, Text, DateTime, ForeignKey, func, Boolean, JSON
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError

# API Configuration
API_KEY = "AIzaSyAG6x7wSSsfheBl5J-bDErcev8-IYA8Cq4"
SEARCH_ENGINE_ID = "0372e57014f804df9"
MAX_SEARCHES_PER_DAY = 100

# Database Configuration
DATABASE_URL = "postgresql://postgres.whwiyccyyfltobvqxiib:SamiHalawa1996@aws-0-eu-central-1.pooler.supabase.com:6543/postgres"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# List of all hospitals
HOSPITALS = [
    "Hospital 12 de Octubre Madrid",
    "Hospital Universitario La Paz Madrid",
    "Hospital Clínico San Carlos Madrid",
    "Hospital Gregorio Marañón Madrid",
    "Hospital Ramón y Cajal Madrid",
    "Hospital Infanta Sofía Madrid",
    "Hospital Puerta de Hierro Majadahonda Madrid",
    "Hospital Universitario Fundación Jiménez Díaz Madrid",
    "Hospital Quirónsalud Madrid",
    "Hospital Sanitas La Moraleja Madrid",
    "Hospital HM Montepríncipe Madrid",
    "Hospital Vithas Madrid"
]

# Database Models
class SearchQuota(Base):
    __tablename__ = 'search_quota'
    id = Column(BigInteger, primary_key=True)
    date = Column(DateTime(timezone=True), unique=True)
    searches_used = Column(BigInteger, default=0)

class HospitalSearch(Base):
    __tablename__ = 'hospital_searches'
    id = Column(BigInteger, primary_key=True)
    hospital_name = Column(Text)
    last_search = Column(DateTime(timezone=True))
    emails_found = Column(BigInteger, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Lead(Base):
    __tablename__ = 'leads'
    id = Column(BigInteger, primary_key=True)
    email = Column(Text, unique=True)
    phone = Column(Text)
    first_name = Column(Text)
    last_name = Column(Text)
    company = Column(Text)
    job_title = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead_sources = relationship("LeadSource", back_populates="lead")

class LeadSource(Base):
    __tablename__ = 'lead_sources'
    id = Column(BigInteger, primary_key=True)
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    url = Column(Text)
    page_title = Column(Text)
    meta_description = Column(Text)
    scrape_duration = Column(Text)
    meta_tags = Column(Text)
    phone_numbers = Column(Text)
    content = Column(Text)
    tags = Column(Text)
    http_status = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead = relationship("Lead", back_populates="lead_sources")

# API Response Models
class SearchResult(BaseModel):
    hospital: str
    emails_found: List[str]
    total_results: int
    sources: List[str]
    is_cached: bool = False

class BatchSearchResult(BaseModel):
    total_hospitals: int
    total_emails: int
    results: List[SearchResult]
    execution_time: float
    searches_remaining: int

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_search_quota(db: Session) -> int:
    today = datetime.now().date()
    quota = db.query(SearchQuota).filter(
        func.date(SearchQuota.date) == today
    ).first()
    
    if not quota:
        quota = SearchQuota(date=today, searches_used=0)
        db.add(quota)
        db.commit()
    
    return MAX_SEARCHES_PER_DAY - quota.searches_used

def increment_search_quota(db: Session):
    today = datetime.now().date()
    quota = db.query(SearchQuota).filter(
        func.date(SearchQuota.date) == today
    ).first()
    
    if quota:
        quota.searches_used += 1
        db.commit()

def get_cached_hospital_results(db: Session, hospital: str) -> tuple[bool, List[str], List[str]]:
    yesterday = datetime.now() - timedelta(days=1)
    hospital_search = db.query(HospitalSearch).filter(
        HospitalSearch.hospital_name == hospital,
        HospitalSearch.last_search > yesterday
    ).first()
    
    if hospital_search:
        leads = db.query(Lead).join(LeadSource).filter(
            LeadSource.url.ilike(f"%{hospital.lower()}%")
        ).all()
        
        sources = db.query(LeadSource.url).join(Lead).filter(
            LeadSource.url.ilike(f"%{hospital.lower()}%")
        ).distinct().all()
        
        return True, [lead.email for lead in leads], [source[0] for source in sources]
    
    return False, [], []

def search_google(query: str, db: Session) -> Dict:
    try:
        if get_search_quota(db) <= 0:
            raise Exception("Daily search quota exceeded")
            
        print(f"\nSearching: {query}")
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "q": query,
            "num": 10
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        increment_search_quota(db)
        print("Search successful")
        return response.json()
    except Exception as e:
        print(f"Search failed: {str(e)}")
        raise

def extract_emails(search_results: Dict) -> tuple[set, List[str]]:
    emails = set()
    sources = []
    
    if 'items' in search_results:
        for item in search_results['items']:
            link = item.get('link', '')
            if link:
                sources.append(link)
            
            if 'snippet' in item:
                snippet = item['snippet']
                found_emails = re.findall(r'[\w\.-]+@[\w\.-]+', snippet)
                for email in found_emails:
                    if email.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        continue
                    emails.add(email.lower())
    
    return emails, sources

def save_to_db(db: Session, email: str, url: str, title: str = "", description: str = "", hospital: str = "") -> bool:
    try:
        existing_lead = db.query(Lead).filter_by(email=email).first()
        if existing_lead:
            print(f"Existing lead: {email}")
            lead_id = existing_lead.id
        else:
            new_lead = Lead(email=email)
            db.add(new_lead)
            db.commit()
            db.refresh(new_lead)
            lead_id = new_lead.id
            print(f"New lead: {email}")

        lead_source = LeadSource(
            lead_id=lead_id,
            url=url,
            page_title=title,
            meta_description=description
        )
        db.add(lead_source)
        
        if hospital:
            hospital_search = db.query(HospitalSearch).filter_by(
                hospital_name=hospital
            ).first()
            
            if hospital_search:
                hospital_search.last_search = datetime.now()
                hospital_search.emails_found += 1
            else:
                hospital_search = HospitalSearch(
                    hospital_name=hospital,
                    last_search=datetime.now(),
                    emails_found=1
                )
                db.add(hospital_search)
        
        db.commit()
        print(f"Saved source: {url}")
        return True
    except Exception as e:
        print(f"Database error: {str(e)}")
        return False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTML Templates
def get_base_html(title: str, content: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container-fluid">
                <a class="navbar-brand" href="/">Email Crawler</a>
                <div class="collapse navbar-collapse">
                    <ul class="navbar-nav me-auto">
                        <li class="nav-item">
                            <a class="nav-link" href="/manual_search">Manual Search</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/view_results">View Results</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/bulk_send">Bulk Send</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/search_terms">Search Terms</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/email_templates">Email Templates</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/knowledge_base">Knowledge Base</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/autoclient_ai">AutoclientAI</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/automation_control">Automation Control</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/manual_search_worker">Manual Search Worker</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/email_logs">Email Logs</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/sent_campaigns">Sent Campaigns</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/settings">Settings</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/projects_campaigns">Projects & Campaigns</a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>
        <div class="container mt-4">
            {content}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """

@app.get("/", response_class=HTMLResponse)
async def home():
    content = """
    <h1>Email Crawler</h1>
    <div class="row mt-4">
        <div class="col-md-6">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Manual Search</h5>
                    <p class="card-text">Search for emails using custom search terms.</p>
                    <a href="/manual_search" class="btn btn-primary">Go to Manual Search</a>
                </div>
            </div>
        </div>
        <div class="col-md-6">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">View Results</h5>
                    <p class="card-text">View all collected email addresses and sources.</p>
                    <a href="/view_results" class="btn btn-primary">View Results</a>
                </div>
            </div>
        </div>
    </div>
    """
    return get_base_html("Email Crawler - Home", content)

@app.get("/manual_search", response_class=HTMLResponse)
async def manual_search_form():
    content = """
    <h2>Manual Search</h2>
    <form method="post" action="/manual_search">
        <div class="mb-3">
            <label for="terms" class="form-label">Search Terms (comma-separated):</label>
            <input type="text" class="form-control" id="terms" name="terms" required>
        </div>
        <div class="mb-3">
            <label for="num_results" class="form-label">Number of Results per Term:</label>
            <input type="number" class="form-control" id="num_results" name="num_results" value="10">
        </div>
        <div class="mb-3">
            <label for="language" class="form-label">Language:</label>
            <input type="text" class="form-control" id="language" name="language" value="ES">
        </div>
        <button type="submit" class="btn btn-primary">Search</button>
    </form>
    """
    return get_base_html("Manual Search", content)

@app.post("/manual_search")
async def do_manual_search(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    terms = data.get("terms", [])
    num_results = data.get("num_results", 10)
    language = data.get("language", "ES")
    
    all_results = []
    total_emails = 0
    
    for term in terms:
        try:
            search_results = search_google(term, db)
            emails, sources = extract_emails(search_results)
            
            for email in emails:
                for source in sources:
                    save_to_db(db, email, source, term)
            
            all_results.append({
                "term": term,
                "emails_found": list(emails),
                "sources": sources,
                "total_results": len(emails)
            })
            
            total_emails += len(emails)
            
        except Exception as e:
            print(f"Error searching term {term}: {str(e)}")
            continue
    
    return {
        "total_emails": total_emails,
        "results": all_results
    }

@app.get("/view_results", response_class=HTMLResponse)
async def view_results(db: Session = Depends(get_db)):
    leads = db.query(Lead).order_by(Lead.created_at.desc()).all()
    
    leads_html = ""
    for lead in leads:
        sources = [source.url for source in lead.lead_sources]
        sources_html = "".join([f'<li><a href="{url}" target="_blank">{url}</a></li>' for url in sources])
        
        leads_html += f"""
        <tr>
            <td>{lead.email}</td>
            <td>{lead.created_at.strftime('%Y-%m-%d %H:%M:%S')}</td>
            <td>
                <button class="btn btn-sm btn-primary" type="button" data-bs-toggle="collapse" 
                        data-bs-target="#sources_{lead.id}">
                    View Sources ({len(sources)})
                </button>
                <div class="collapse" id="sources_{lead.id}">
                    <ul class="mt-2">
                        {sources_html}
                    </ul>
                </div>
            </td>
        </tr>
        """
    
    content = f"""
    <h2>Search Results</h2>
    <div class="table-responsive mt-4">
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>Email</th>
                    <th>Found At</th>
                    <th>Sources</th>
                </tr>
            </thead>
            <tbody>
                {leads_html}
            </tbody>
        </table>
    </div>
    """
    
    return get_base_html("View Results", content)

if __name__ == "__main__":
    import uvicorn
    Base.metadata.create_all(bind=engine)
    uvicorn.run(app, host="0.0.0.0", port=8000)