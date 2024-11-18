import { Button } from "./button"
import { ScrollArea } from "./scroll-area"
import { cn } from "@/lib/utils"
import { 
  Search, 
  Mail, 
  Users, 
  Key, 
  FileText,
  Folder,
  Book,
  Robot,
  Settings,
  List,
  Send
} from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"

const sidebarItems = [
  {
    title: "Manual Search",
    href: "/manual-search",
    icon: Search
  },
  {
    title: "Bulk Send",
    href: "/bulk-send", 
    icon: Send
  },
  {
    title: "View Leads",
    href: "/view-leads",
    icon: Users
  },
  {
    title: "Search Terms",
    href: "/search-terms",
    icon: Key
  },
  {
    title: "Email Templates",
    href: "/email-templates",
    icon: Mail
  },
  {
    title: "Projects & Campaigns",
    href: "/projects",
    icon: Folder
  },
  {
    title: "Knowledge Base",
    href: "/knowledge-base",
    icon: Book
  },
  {
    title: "AutoclientAI",
    href: "/autoclient-ai",
    icon: Robot
  },
  {
    title: "Automation Control",
    href: "/automation",
    icon: Settings
  },
  {
    title: "Email Logs",
    href: "/email-logs",
    icon: List
  },
  {
    title: "Settings",
    href: "/settings",
    icon: Settings
  },
  {
    title: "Sent Campaigns",
    href: "/sent-campaigns",
    icon: Mail
  }
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <div className="pb-12 w-64">
      <div className="space-y-4 py-4">
        <div className="px-3 py-2">
          <h2 className="mb-2 px-4 text-lg font-semibold">Navigation</h2>
          <ScrollArea className="h-[calc(100vh-10rem)]">
            <div className="space-y-1">
              {sidebarItems.map((item) => (
                <Link key={item.href} href={item.href}>
                  <Button
                    variant={pathname === item.href ? "secondary" : "ghost"}
                    className={cn(
                      "w-full justify-start",
                      pathname === item.href && "bg-muted"
                    )}
                  >
                    <item.icon className="mr-2 h-4 w-4" />
                    {item.title}
                  </Button>
                </Link>
              ))}
            </div>
          </ScrollArea>
        </div>
      </div>
    </div>
  )
} 