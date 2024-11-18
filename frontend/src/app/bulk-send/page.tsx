"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"
import { DataTable } from "@/components/ui/data-table"

interface Lead {
  id: string
  email: string
  status: string
}

export default function BulkSend() {
  const [sendOption, setSendOption] = useState("all")
  const [specificEmail, setSpecificEmail] = useState("")
  const [selectedTerms, setSelectedTerms] = useState<string[]>([])
  const [excludePreviouslyContacted, setExcludePreviouslyContacted] = useState(true)
  const [progress, setProgress] = useState(0)
  const [isSending, setIsSending] = useState(false)
  const [leads, setLeads] = useState<Lead[]>([])

  const handleSend = async () => {
    setIsSending(true)
    setProgress(0)

    try {
      const response = await fetch("/api/bulk-send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sendOption,
          specificEmail,
          selectedTerms,
          excludePreviouslyContacted
        })
      })
      
      const data = await response.json()
      setLeads(data.leads)
    } catch (error) {
      console.error("Bulk send failed:", error)
    } finally {
      setIsSending(false)
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Bulk Send</h1>

      <Card>
        <CardHeader>
          <CardTitle>Email Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <Select>
              <SelectTrigger>
                <SelectValue placeholder="Select email template" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="template1">Template 1</SelectItem>
                <SelectItem value="template2">Template 2</SelectItem>
              </SelectContent>
            </Select>

            <div className="grid gap-4">
              <Select>
                <SelectTrigger>
                  <SelectValue placeholder="From email" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="email1">email1@example.com</SelectItem>
                  <SelectItem value="email2">email2@example.com</SelectItem>
                </SelectContent>
              </Select>

              <Input placeholder="Reply-to email address" />
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-medium">Send Options</h3>
            <RadioGroup value={sendOption} onValueChange={setSendOption}>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="all" id="all" />
                <Label htmlFor="all">All Leads</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="specific" id="specific" />
                <Label htmlFor="specific">Specific Email</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="terms" id="terms" />
                <Label htmlFor="terms">Leads from Search Terms</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="groups" id="groups" />
                <Label htmlFor="groups">Leads from Search Term Groups</Label>
              </div>
            </RadioGroup>

            {sendOption === "specific" && (
              <Input 
                placeholder="Enter email address" 
                value={specificEmail}
                onChange={(e) => setSpecificEmail(e.target.value)}
              />
            )}

            {(sendOption === "terms" || sendOption === "groups") && (
              <Select>
                <SelectTrigger>
                  <SelectValue placeholder="Select terms or groups" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="term1">Term 1</SelectItem>
                  <SelectItem value="term2">Term 2</SelectItem>
                </SelectContent>
              </Select>
            )}

            <div className="flex items-center space-x-2">
              <Checkbox 
                id="exclude" 
                checked={excludePreviouslyContacted}
                onCheckedChange={(checked) => setExcludePreviouslyContacted(checked as boolean)}
              />
              <Label htmlFor="exclude">Exclude Previously Contacted</Label>
            </div>
          </div>

          <Button 
            className="w-full" 
            size="lg"
            onClick={handleSend}
            disabled={isSending}
          >
            {isSending ? "Sending..." : "Send Emails"}
          </Button>
        </CardContent>
      </Card>

      {isSending && (
        <Card>
          <CardContent className="pt-6">
            <Progress value={progress} className="mb-2" />
            <p className="text-center text-sm text-muted-foreground">
              Sending emails... {progress}% complete
            </p>
          </CardContent>
        </Card>
      )}

      {leads.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Email Status</CardTitle>
          </CardHeader>
          <CardContent>
            <DataTable 
              columns={[
                { accessorKey: "email", header: "Email" },
                { accessorKey: "status", header: "Status" }
              ]}
              data={leads}
            />
          </CardContent>
        </Card>
      )}
    </div>
  )
} 