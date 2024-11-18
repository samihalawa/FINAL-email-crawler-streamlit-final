"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Progress } from "@/components/ui/progress"
import { DataTable } from "@/components/ui/data-table"

interface SearchResult {
  email: string
  url: string
  title: string
  company: string
  source: string
}

export default function ManualSearch() {
  const [searchTerms, setSearchTerms] = useState<string[]>([])
  const [numResults, setNumResults] = useState(10)
  const [enableEmailSending, setEnableEmailSending] = useState(true)
  const [progress, setProgress] = useState(0)
  const [results, setResults] = useState<SearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)

  const handleSearch = async () => {
    setIsSearching(true)
    setProgress(0)
    
    try {
      // Call your API endpoint here
      const response = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          terms: searchTerms,
          numResults,
          enableEmailSending
        })
      })
      
      const data = await response.json()
      setResults(data.results)
    } catch (error) {
      console.error("Search failed:", error)
    } finally {
      setIsSearching(false)
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Manual Search</h1>

      <Card>
        <CardHeader>
          <CardTitle>Search Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label>Search Terms</label>
            <Input 
              placeholder="Enter search terms (press Enter to add)"
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  const input = e.currentTarget
                  if (input.value.trim()) {
                    setSearchTerms([...searchTerms, input.value.trim()])
                    input.value = ""
                  }
                }
              }}
            />
            <div className="flex flex-wrap gap-2">
              {searchTerms.map((term, i) => (
                <Button
                  key={i}
                  variant="secondary"
                  size="sm"
                  onClick={() => setSearchTerms(searchTerms.filter((_, idx) => idx !== i))}
                >
                  {term} ×
                </Button>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <label>Results per term: {numResults}</label>
            <Slider
              value={[numResults]}
              onValueChange={([value]) => setNumResults(value)}
              min={1}
              max={500}
              step={1}
            />
          </div>

          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Switch
                checked={enableEmailSending}
                onCheckedChange={setEnableEmailSending}
              />
              <label>Enable email sending</label>
            </div>

            {enableEmailSending && (
              <>
                <Select>
                  <SelectTrigger>
                    <SelectValue placeholder="Select email template" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="template1">Template 1</SelectItem>
                    <SelectItem value="template2">Template 2</SelectItem>
                  </SelectContent>
                </Select>

                <Select>
                  <SelectTrigger>
                    <SelectValue placeholder="Select from email" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="email1">email1@example.com</SelectItem>
                    <SelectItem value="email2">email2@example.com</SelectItem>
                  </SelectContent>
                </Select>

                <Input placeholder="Reply-to email address" />
              </>
            )}
          </div>

          <Button 
            className="w-full" 
            size="lg"
            onClick={handleSearch}
            disabled={isSearching || searchTerms.length === 0}
          >
            {isSearching ? "Searching..." : "Start Search"}
          </Button>
        </CardContent>
      </Card>

      {isSearching && (
        <Card>
          <CardContent className="pt-6">
            <Progress value={progress} className="mb-2" />
            <p className="text-center text-sm text-muted-foreground">
              Searching... {progress}% complete
            </p>
          </CardContent>
        </Card>
      )}

      {results.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Search Results</CardTitle>
          </CardHeader>
          <CardContent>
            <DataTable 
              columns={[
                { accessorKey: "email", header: "Email" },
                { accessorKey: "url", header: "URL" },
                { accessorKey: "title", header: "Title" },
                { accessorKey: "company", header: "Company" },
                { accessorKey: "source", header: "Source" }
              ]}
              data={results}
            />
          </CardContent>
        </Card>
      )}
    </div>
  )
} 