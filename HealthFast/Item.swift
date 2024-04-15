//
//  Item.swift
//  HealthFast
//
//  Created by Kousthubh Veturi on 4/9/24.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
