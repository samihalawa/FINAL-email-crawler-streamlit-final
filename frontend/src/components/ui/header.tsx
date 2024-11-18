import { ModeToggle } from "./mode-toggle"
import { UserNav } from "./user-nav"
import { MainNav } from "./main-nav"

export function Header() {
  return (
    <header className="border-b">
      <div className="flex h-16 items-center px-4">
        <MainNav />
        <div className="ml-auto flex items-center space-x-4">
          <ModeToggle />
          <UserNav />
        </div>
      </div>
    </header>
  )
} 